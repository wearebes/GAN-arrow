"""Minimal SDXL LoRA trainer for the target-face dataset.

Design constraints for a fanless 24 GB M3 MacBook Air:
- UNet only, fp32 weights loaded from the fp16 variant files, gradient
  checkpointing on, batch 1 with gradient accumulation.
- VAE and text encoders are never loaded: latents and prompt embeddings come
  from the cache produced by precompute_latents.py.
- Everything except the LoRA adapter (attention to_q/to_k/to_v/to_out.0) is
  frozen.

Validation loss is computed on the 8 held-out images with a fixed
(timestep, noise) grid so values are comparable across checkpoints, and the
val images use the latent-distribution mean (no sampling noise).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VAL_TIMESTEPS = [100, 300, 500, 700, 900]


def sample_latent(params: torch.Tensor, scaling: float, generator=None) -> torch.Tensor:
    mean, logvar = params.chunk(2, dim=1)
    std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
    noise = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
    return (mean + std * noise) * scaling


def mean_latent(params: torch.Tensor, scaling: float) -> torch.Tensor:
    mean, _ = params.chunk(2, dim=1)
    return mean * scaling


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="outputs/cache directory")
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--init-lora", default=None, help="raw_peft.pt to resume from")
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16", "autocast"], default="fp16",
                    help="fp32: full precision; fp16/bf16: whole frozen base in that dtype "
                         "(LoRA + loss stay fp32); autocast: fp32 weights with bf16 autocast")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--no-initial-val", action="store_true")
    args = ap.parse_args()

    from diffusers import DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

    run_dir = Path(args.run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    cache = Path(args.cache)
    train_cache = torch.load(cache / f"train_{args.res}.pt", map_location="cpu")
    val_cache = torch.load(cache / f"val_{args.res}.pt", map_location="cpu")
    text = torch.load(cache / "text_embeds.pt", map_location="cpu")
    scaling = float(train_cache["scaling_factor"])

    train_params = train_cache["latent_params"]  # [N, 8, h, w]
    val_params = val_cache["latent_params"]
    n_train, n_val = train_params.shape[0], val_params.shape[0]

    model_dtype = {"fp32": torch.float32, "fp16": torch.float16,
                   "bf16": torch.bfloat16, "autocast": torch.float32}[args.precision]
    use_autocast = args.precision == "autocast"

    def fwd_ctx():
        if use_autocast:
            return torch.autocast(device_type="mps", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    prompt_embeds = text["prompt_embeds"].to(device, model_dtype)  # [1, 77, 2048]
    pooled_embeds = text["pooled_embeds"].to(device, model_dtype)  # [1, 1280]
    time_ids = torch.tensor(
        [[args.res, args.res, 0, 0, args.res, args.res]],
        device=device, dtype=model_dtype,
    )

    print(f"train entries {n_train} (with flips), val {n_val}, res {args.res}", flush=True)

    unet = UNet2DConditionModel.from_pretrained(
        MODEL, subfolder="unet", variant="fp16", torch_dtype=model_dtype
    ).to(device)
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    if args.init_lora:
        state = torch.load(args.init_lora, map_location="cpu")
        set_peft_model_state_dict(unet, state)
        print(f"resumed LoRA weights from {args.init_lora}", flush=True)
    # LoRA params stay fp32 regardless of base dtype (peft casts activations);
    # keeps AdamW states in full precision.
    for p in unet.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    if not args.no_grad_ckpt:
        unet.enable_gradient_checkpointing()
    unet.train()

    trainable = [p for p in unet.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in unet.parameters())
    print(f"trainable params: {n_trainable/1e6:.2f}M / {n_total/1e6:.0f}M", flush=True)

    noise_sched = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler")
    n_timesteps = noise_sched.config.num_train_timesteps

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )

    # Fixed validation grid: latent means + one fixed noise per (image, timestep).
    gen = torch.Generator().manual_seed(1234)
    val_latents = mean_latent(val_params, scaling)  # CPU [n_val, 4, h, w]
    val_noise = torch.randn(
        (n_val, len(VAL_TIMESTEPS)) + tuple(val_latents.shape[1:]), generator=gen
    )

    def validate() -> tuple[float, list[float]]:
        unet.eval()
        per_t = []
        with torch.no_grad():
            for ti, t_val in enumerate(VAL_TIMESTEPS):
                losses = []
                for vi in range(n_val):
                    z = val_latents[vi : vi + 1].to(device, model_dtype)
                    noise = val_noise[vi, ti : ti + 1].to(device, model_dtype)
                    t = torch.tensor([t_val], device=device, dtype=torch.long)
                    noisy = noise_sched.add_noise(z, noise, t)
                    with fwd_ctx():
                        pred = unet(
                            noisy, t,
                            encoder_hidden_states=prompt_embeds,
                            added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids},
                        ).sample
                    losses.append(F.mse_loss(pred.float(), noise.float()).item())
                per_t.append(sum(losses) / len(losses))
        unet.train()
        return sum(per_t) / len(per_t), per_t

    def save_ckpt(step: int) -> None:
        d = run_dir / "checkpoints" / f"step_{step:04d}"
        d.mkdir(parents=True, exist_ok=True)
        peft_sd = get_peft_model_state_dict(unet)
        torch.save(peft_sd, d / "raw_peft.pt")
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=str(d),
            unet_lora_layers=convert_state_dict_to_diffusers(peft_sd),
            safe_serialization=True,
        )
        print(f"saved checkpoint {d.name}", flush=True)

    (run_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "trainable_params": n_trainable,
                    "train_entries": n_train, "val_images": n_val,
                    "val_timesteps": VAL_TIMESTEPS, "model": MODEL,
                    "prompt": text["prompt"]}, indent=2)
    )

    train_log = open(run_dir / "train_log.csv", "a", newline="")
    train_writer = csv.writer(train_log)
    val_log = open(run_dir / "val_log.csv", "a", newline="")
    val_writer = csv.writer(val_log)
    if args.start_step == 0:
        train_writer.writerow(["step", "loss", "lr", "sec_per_step", "mps_gb"])
        val_writer.writerow(["step", "val_loss"] + [f"t{t}" for t in VAL_TIMESTEPS])

    if not args.no_initial_val:
        v0, v0_per_t = validate()
        val_writer.writerow([args.start_step, f"{v0:.6f}"] + [f"{x:.6f}" for x in v0_per_t])
        val_log.flush()
        print(f"[val] step {args.start_step}: {v0:.5f} per-t {[round(x,4) for x in v0_per_t]}", flush=True)

    perm = torch.randperm(n_train)
    cursor = 0

    def next_indices(k: int) -> torch.Tensor:
        nonlocal perm, cursor
        out = []
        while len(out) < k:
            if cursor >= n_train:
                perm = torch.randperm(n_train)
                cursor = 0
            out.append(perm[cursor].item())
            cursor += 1
        return torch.tensor(out)

    t_start = time.time()
    for step in range(args.start_step + 1, args.steps + 1):
        step_t0 = time.time()
        lr_scale = min(1.0, step / max(args.warmup, 1))
        for g in optimizer.param_groups:
            g["lr"] = args.lr * lr_scale

        accum_loss = 0.0
        for _ in range(args.accum):
            idx = next_indices(args.batch)
            z = sample_latent(train_params[idx].to(device), scaling).to(model_dtype)
            noise = torch.randn_like(z)
            t = torch.randint(0, n_timesteps, (z.shape[0],), device=device)
            noisy = noise_sched.add_noise(z, noise, t)
            with fwd_ctx():
                pred = unet(
                    noisy, t,
                    encoder_hidden_states=prompt_embeds.expand(z.shape[0], -1, -1),
                    added_cond_kwargs={
                        "text_embeds": pooled_embeds.expand(z.shape[0], -1),
                        "time_ids": time_ids.expand(z.shape[0], -1),
                    },
                ).sample
                loss = F.mse_loss(pred.float(), noise.float()) / args.accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        sec = time.time() - step_t0
        mem = torch.mps.current_allocated_memory() / 1e9 if device.type == "mps" else 0.0
        train_writer.writerow([step, f"{accum_loss:.6f}", f"{args.lr * lr_scale:.2e}",
                               f"{sec:.2f}", f"{mem:.2f}"])
        train_log.flush()

        if step % 10 == 0 or step == args.start_step + 1:
            done = step - args.start_step
            eta_h = (args.steps - step) * (time.time() - t_start) / done / 3600
            print(f"step {step}/{args.steps} loss {accum_loss:.4f} "
                  f"{sec:.1f}s/step mem {mem:.1f}GB eta {eta_h:.2f}h", flush=True)

        if step % args.val_every == 0 or step == args.steps:
            v, per_t = validate()
            val_writer.writerow([step, f"{v:.6f}"] + [f"{x:.6f}" for x in per_t])
            val_log.flush()
            print(f"[val] step {step}: {v:.5f} per-t {[round(x,4) for x in per_t]}", flush=True)

        if step % args.ckpt_every == 0 or step == args.steps:
            save_ckpt(step)

        if device.type == "mps" and step % 100 == 0:
            torch.mps.empty_cache()

    train_log.close()
    val_log.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
