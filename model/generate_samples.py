import argparse
import json
from pathlib import Path

import torch
import torchvision.utils as vutils

from model.train_gan import Generator, select_device


def _checkpoint_config(checkpoint):
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    return {
        "image_size": int(config.get("image_size", 256)),
        "latent_dim": int(config.get("latent_dim", 100)),
        "channels": int(config.get("channels", 3)),
        "generator_features": int(config.get("generator_features", config.get("features", 32))),
    }


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_generator(checkpoint_path: Path, device=None):
    device = device or select_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = _checkpoint_config(checkpoint)
    generator = Generator(
        image_size=config["image_size"],
        latent_dim=config["latent_dim"],
        channels=config["channels"],
        features=config["generator_features"],
    ).to(device)
    generator.load_state_dict(_checkpoint_state_dict(checkpoint))
    generator.eval()
    return generator, config


def generate_samples(checkpoint_path: Path, output_dir: Path, num_samples=64, seed=42, batch_size=16):
    device = select_device()
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    generator, config = load_generator(checkpoint_path, device=device)
    all_images = []
    generated_count = 0
    with torch.no_grad():
        while generated_count < num_samples:
            current_batch = min(batch_size, num_samples - generated_count)
            noise = torch.randn(current_batch, config["latent_dim"], 1, 1, device=device)
            images = generator(noise).detach().cpu()
            for image in images:
                generated_count += 1
                sample_path = samples_dir / f"sample_{generated_count:03d}.png"
                vutils.save_image(image, sample_path, normalize=True)
                all_images.append(image)

    contact_sheet_path = output_dir / "contact_sheet.png"
    vutils.save_image(torch.stack(all_images), contact_sheet_path, normalize=True, nrow=8)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "generated_count": generated_count,
        "output_dir": str(output_dir),
        "samples_dir": str(samples_dir),
        "contact_sheet": str(contact_sheet_path),
        "config": config,
    }
    with (output_dir / "generation_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Generate samples from a saved GAN generator checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    metrics = generate_samples(
        checkpoint_path=args.checkpoint,
        output_dir=args.out,
        num_samples=args.num,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
