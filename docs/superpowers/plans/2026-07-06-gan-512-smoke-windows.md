# 512 GAN Smoke and Windows Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 512px GAN training path that can be smoke-tested locally, then run the formal generator-strengthening experiment on the user's Windows machine with 8GB VRAM and 32GB RAM.

**Architecture:** Keep the task as whole ROI image generation, not patch generation. Use 512px full target ROI images, strengthen the generator by moving from the current 256 G64/D32 baseline to 512 G96/D32, and add training infrastructure needed for 8GB VRAM: AMP, gradient accumulation, checkpoint/sample intervals, and discriminator-probability evaluation.

**Tech Stack:** Python, PyTorch, torchvision, PIL, existing `model/train_gan.py`, existing `model/generate_samples.py`, existing `model/score_target.py`, YAML configs.

---

## Current Baseline

The current preserved best run is:

- `image_size=256`
- `processed_dir=dataset/generate_data/processed_256_front`
- `generator_mode=upsample`
- `generator_features=64`
- `discriminator_mode=global`
- `discriminator_features=32`
- `batch_size=16`
- `d_lr=0.0001`
- `g_lr=0.0002`
- `diffaugment=color,translation,cutout`
- `ema_decay=0.999`
- preserved checkpoint: `outputs/L256_003_g64_d32/gan_20260706_022859/best_generator.pt`

The next step is not a long local run. The local goal is only to prove that a 512 training path works end-to-end. The formal run happens on Windows.

## Scope

Do:

- Add 512 configuration files.
- Add AMP support.
- Add gradient accumulation.
- Add interval-based sample saving and rolling recovery checkpoints to avoid per-epoch output bloat.
- Add a discriminator probability evaluator so generator progress is judged by the project's own discriminator.
- Run only a smoke test locally.
- Prepare Windows commands for the formal experiment.

Do not:

- Start a long local 512 experiment.
- Delete the current 256 best checkpoint.
- Switch to patch generation.
- Move to 1024 in this phase.
- Add a residual generator in the first 512 smoke path. Residual upsampling is reserved for the next architecture step if 512 G96 remains too smooth or gets very low D-score.

## File Map

- Modify `model/train_gan.py`
  - Add `amp`, `grad_accum_steps`, `sample_interval`, `checkpoint_interval`, and `freeze_discriminator_during_generator_step` to `TrainingConfig`.
  - Add CLI flags for those options.
  - Use AMP on CUDA when enabled.
  - Accumulate gradients over several mini-batches.
  - Freeze D parameters during the G update so G can receive gradients through D without storing D parameter gradients.
  - Save sample previews by interval and refresh a rolling recovery checkpoint by interval, not every epoch.

- Create `model/evaluate_discriminator.py`
  - Load a discriminator checkpoint.
  - Evaluate `P(real)=sigmoid(D(x))` on real ROI images.
  - Evaluate `P(real)` on saved generated PNG samples.
  - Optionally evaluate fresh samples from a generator checkpoint.
  - Write JSON summary with mean, median, min, max, quartiles, and 0.5-threshold counts.

- Create `configs/gan_512_smoke.yaml`
  - Local smoke configuration.

- Create `configs/gan_512_win_formal.yaml`
  - Windows formal training configuration.

- Modify `tests/test_train_gan.py`
  - Cover config fields.
  - Cover gradient accumulation loss scaling.
  - Cover D-freeze behavior during G update.
  - Cover discriminator evaluator summary logic.

## Task 1: Add 512 Training Controls

**Files:**
- Modify: `model/train_gan.py`
- Modify: `tests/test_train_gan.py`

- [ ] **Step 1: Add failing config tests**

Add tests that instantiate:

```python
config = TrainingConfig.from_dataset_size(
    109,
    image_size=512,
    processed_dir=Path("dataset/generate_data/processed_512_front"),
    generator_features=96,
    discriminator_features=32,
    batch_size=4,
    amp=True,
    grad_accum_steps=4,
    sample_interval=25,
    checkpoint_interval=25,
    freeze_discriminator_during_generator_step=True,
)
```

Expected assertions:

- `config.image_size == 512`
- `config.generator_features == 96`
- `config.discriminator_features == 32`
- `config.batch_size == 4`
- `config.amp is True`
- `config.grad_accum_steps == 4`
- `config.sample_interval == 25`
- `config.checkpoint_interval == 25`
- `config.freeze_discriminator_during_generator_step is True`

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
/opt/anaconda3/envs/gan/bin/python -m unittest tests.test_train_gan.GanTrainingConfigTests.test_512_training_config_can_enable_memory_controls
```

Expected: fail because the new fields do not exist yet.

- [ ] **Step 3: Add `TrainingConfig` fields**

Add defaults:

```python
amp: bool = False
grad_accum_steps: int = 1
sample_interval: int = 1
checkpoint_interval: int = 1
freeze_discriminator_during_generator_step: bool = True
```

Validate in `TrainingConfig.from_dataset_size` or train startup:

- `grad_accum_steps >= 1`
- `sample_interval >= 1`
- `checkpoint_interval >= 1`

- [ ] **Step 4: Add CLI flags**

Add:

```text
--amp
--grad-accum-steps
--sample-interval
--checkpoint-interval
--no-freeze-discriminator-during-generator-step
```

Map them into `overrides` in `main()`.

- [ ] **Step 5: Re-run config test**

Run:

```bash
/opt/anaconda3/envs/gan/bin/python -m unittest tests.test_train_gan.GanTrainingConfigTests.test_512_training_config_can_enable_memory_controls
```

Expected: pass.

## Task 2: Implement AMP and Gradient Accumulation

**Files:**
- Modify: `model/train_gan.py`
- Modify: `tests/test_train_gan.py`

- [ ] **Step 1: Add small helper tests**

Add testable helpers:

```python
def should_use_amp(config: TrainingConfig, device: torch.device) -> bool:
    return bool(config.amp and device.type == "cuda")

def scale_loss_for_accumulation(loss, grad_accum_steps: int):
    return loss / grad_accum_steps
```

Tests:

- CUDA device + `amp=True` returns `True`.
- CPU device + `amp=True` returns `False`.
- `scale_loss_for_accumulation(torch.tensor(8.0), 4)` returns `2.0`.

- [ ] **Step 2: Run helper tests and verify they fail**

Run:

```bash
/opt/anaconda3/envs/gan/bin/python -m unittest tests.test_train_gan.GanTrainingConfigTests.test_amp_only_enables_on_cuda tests.test_train_gan.GanTrainingConfigTests.test_gradient_accumulation_scales_loss
```

Expected: fail because helpers do not exist.

- [ ] **Step 3: Implement helpers and AMP scaffolding**

Use:

```python
use_amp = should_use_amp(config, device)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
```

For PyTorch versions that prefer the newer API, keep compatibility by using the available `torch.cuda.amp` API already supported by common Windows CUDA installs.

- [ ] **Step 4: Update D and G training loops**

Accumulate for `config.grad_accum_steps` mini-batches:

- Divide D loss by `grad_accum_steps` before backward.
- Divide G loss by `grad_accum_steps` before backward.
- Call optimizer step only when the accumulation boundary is reached or the dataloader is exhausted.
- Zero gradients only after optimizer step.
- Log unscaled losses for readable metrics.

- [ ] **Step 5: Freeze D parameters during G update**

During G update:

```python
if config.freeze_discriminator_during_generator_step:
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)

output_for_generator = discriminator(fake_for_generator)

if config.freeze_discriminator_during_generator_step:
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
```

This keeps gradients flowing from D output to fake image, but avoids D parameter gradients during G optimization.

- [ ] **Step 6: Run focused tests**

Run:

```bash
/opt/anaconda3/envs/gan/bin/python -m unittest tests.test_train_gan
```

Expected: all tests pass.

## Task 3: Add Interval-Based Artifact Saving

**Files:**
- Modify: `model/train_gan.py`
- Modify: `tests/test_train_gan.py`

- [ ] **Step 1: Add helper tests**

Add:

```python
def should_save_epoch_artifact(epoch: int, total_epochs: int, interval: int) -> bool:
    return epoch == 1 or epoch == total_epochs or epoch % interval == 0
```

Expected behavior:

- epoch 1 saves.
- final epoch saves.
- epoch 25 saves with interval 25.
- epoch 24 does not save with interval 25.

- [ ] **Step 2: Apply helper in training**

Use `sample_interval` for `samples_epoch_*.png`.

Use `checkpoint_interval` to refresh `training_state_latest.pt` as a rolling recovery checkpoint. Do not write a separate numbered full checkpoint every interval in this phase; a 3000-epoch Windows run would create too many large files.

Always save final:

- `generator.pt`
- `generator_ema.pt` when EMA is enabled
- `discriminator.pt`
- `best_generator.pt`
- `training_state.pt`
- `training_state_latest.pt`
- `metrics.json`
- `history.csv`
- `loss_curve.png`

- [ ] **Step 3: Run focused tests**

Run:

```bash
/opt/anaconda3/envs/gan/bin/python -m unittest tests.test_train_gan.GanTrainingConfigTests.test_should_save_epoch_artifact_respects_interval
```

Expected: pass.

## Task 4: Add Discriminator Probability Evaluation

**Files:**
- Create: `model/evaluate_discriminator.py`
- Modify: `tests/test_train_gan.py`

- [ ] **Step 1: Add summary helper**

Create:

```python
def summarize_probabilities(probabilities, fake_expected: bool):
    ...
```

The return object must include:

- `count`
- `mean_p_real`
- `median_p_real`
- `min_p_real`
- `q25_p_real`
- `q75_p_real`
- `max_p_real`
- `correct_at_0_5`
- `accuracy_at_0_5`

For real images, correct means `p >= 0.5`.

For generated images, correct means `p < 0.5`.

- [ ] **Step 2: Add unit test for summary helper**

Use real probabilities `[0.8, 0.9, 0.95]` and generated probabilities `[0.1, 0.2, 0.4]`.

Expected:

- real `mean_p_real == 0.8833333333333333`
- generated `mean_p_real == 0.23333333333333334`
- both accuracies equal `1.0`.

- [ ] **Step 3: Implement CLI**

CLI arguments:

```text
--discriminator-checkpoint
--metrics
--real-dir
--generated-dir
--generator-checkpoint
--num-fresh
--seed
--out
--batch-size
```

The `--metrics` file provides image size, discriminator features, discriminator mode, norm, channels, and generator config. If `--metrics` is omitted, require explicit model options.

- [ ] **Step 4: Run evaluator on preserved 256 checkpoint**

Run:

```bash
/opt/anaconda3/envs/gan/bin/python -m model.evaluate_discriminator \
  --discriminator-checkpoint outputs/L256_003_g64_d32/gan_20260706_022859/discriminator.pt \
  --metrics outputs/L256_003_g64_d32/gan_20260706_022859/metrics.json \
  --real-dir dataset/generate_data/processed_256_front \
  --generated-dir outputs/L256_003_g64_d32/generated_seg4/samples \
  --generator-checkpoint outputs/L256_003_g64_d32/gan_20260706_022859/best_generator.pt \
  --num-fresh 64 \
  --out outputs/L256_003_g64_d32/discriminator_eval_seg4.json
```

Expected:

- JSON file is written.
- Real mean `P(real)` is high.
- Generated mean `P(real)` is low.

## Task 5: Add 512 Configs

**Files:**
- Create: `configs/gan_512_smoke.yaml`
- Create: `configs/gan_512_win_formal.yaml`

- [ ] **Step 1: Create local smoke config**

`configs/gan_512_smoke.yaml`:

```yaml
image_size: 512
processed_dir: dataset/generate_data/processed_512_front
output_dir: outputs/smoke_512_g96_d32
epochs: 1
max_steps: 1
batch_size: 1
generator_mode: upsample
generator_features: 96
discriminator_mode: global
discriminator_features: 32
discriminator_norm: none
adversarial_loss_mode: bce
d_lr: 0.0001
g_lr: 0.0002
real_label: 0.9
diffaugment: true
diffaugment_policy: color,translation,cutout
ema_decay: 0.999
target_prior_weight: 0.0
amp: false
grad_accum_steps: 1
sample_interval: 1
checkpoint_interval: 1
skip_prepare: true
min_target_anchor_fraction: 0.01
target_crop_expansion: 2.9
seed: 52
```

Local smoke uses `batch_size=1` and `skip_prepare=true` because the local machine may use CPU and HEIC preprocessing is slow. Windows formal uses `batch_size=4` and rebuilds the 512 cache from source unless the user explicitly copies a prepared cache.

- [ ] **Step 2: Create Windows formal config**

`configs/gan_512_win_formal.yaml`:

```yaml
image_size: 512
processed_dir: dataset/generate_data/processed_512_front
output_dir: outputs/W512_001_g96_d32
epochs: 3000
max_steps: null
batch_size: 4
generator_mode: upsample
generator_features: 96
discriminator_mode: global
discriminator_features: 32
discriminator_norm: none
adversarial_loss_mode: bce
d_lr: 0.0001
g_lr: 0.0002
real_label: 0.9
diffaugment: true
diffaugment_policy: color,translation,cutout
ema_decay: 0.999
target_prior_weight: 0.0
amp: true
grad_accum_steps: 4
sample_interval: 50
checkpoint_interval: 100
skip_prepare: false
min_target_anchor_fraction: 0.01
target_crop_expansion: 2.9
seed: 52
```

Effective batch is `batch_size * grad_accum_steps = 16`.

## Task 6: Run Local 512 Smoke Only

**Files:**
- No code edits after this task unless smoke reveals a defect.

- [ ] **Step 1: Run local smoke training**

Run:

```bash
/opt/anaconda3/envs/gan/bin/python -m model.train_gan \
  --image-size 512 \
  --processed-dir dataset/generate_data/processed_512_front \
  --output-dir outputs/smoke_512_g96_d32 \
  --epochs 1 \
  --max-steps 1 \
  --batch-size 1 \
  --generator-features 96 \
  --discriminator-features 32 \
  --discriminator-norm none \
  --discriminator-mode global \
  --adversarial-loss-mode bce \
  --d-lr 0.0001 \
  --g-lr 0.0002 \
  --diffaugment \
  --diffaugment-policy color,translation,cutout \
  --ema-decay 0.999 \
  --target-prior-weight 0.0 \
  --grad-accum-steps 1 \
  --sample-interval 1 \
  --checkpoint-interval 1 \
  --skip-prepare \
  --seed 52
```

Expected:

- One run directory appears under `outputs/smoke_512_g96_d32/`.
- `metrics.json` is written.
- `generator.pt`, `discriminator.pt`, `best_generator.pt`, `training_state.pt`, and `samples_epoch_001.png` are written.
- `metrics.json` reports `image_size=512`, `generator_features=96`, and `discriminator_features=32`.

- [ ] **Step 2: Generate smoke samples from the smoke checkpoint**

Run:

```bash
RUN_DIR=$(find outputs/smoke_512_g96_d32 -maxdepth 1 -type d -name 'gan_*' | sort | tail -n 1)
/opt/anaconda3/envs/gan/bin/python -m model.generate_samples \
  --checkpoint "$RUN_DIR/best_generator.pt" \
  --num 16 \
  --out outputs/smoke_512_g96_d32/generated_check \
  --seed 52 \
  --batch-size 4
```

Expected:

- `outputs/smoke_512_g96_d32/generated_check/contact_sheet.png` exists.
- `outputs/smoke_512_g96_d32/generated_check/generation_metrics.json` reports `image_size=512`.

- [ ] **Step 3: Run D-probability smoke evaluation**

Run:

```bash
RUN_DIR=$(find outputs/smoke_512_g96_d32 -maxdepth 1 -type d -name 'gan_*' | sort | tail -n 1)
/opt/anaconda3/envs/gan/bin/python -m model.evaluate_discriminator \
  --discriminator-checkpoint "$RUN_DIR/discriminator.pt" \
  --metrics "$RUN_DIR/metrics.json" \
  --real-dir dataset/generate_data/processed_512_front \
  --generated-dir outputs/smoke_512_g96_d32/generated_check/samples \
  --generator-checkpoint "$RUN_DIR/best_generator.pt" \
  --num-fresh 16 \
  --out outputs/smoke_512_g96_d32/discriminator_eval_smoke.json \
  --batch-size 4
```

Expected:

- JSON output exists.
- It includes real, saved generated, and fresh generated summaries.
- The purpose is only to prove the metric path works. Smoke D-score does not decide model quality.

## Task 7: Windows Formal Run

**Files:**
- No local code edits.

- [ ] **Step 1: Prepare Windows environment**

Use the same repository copy and install a CUDA PyTorch environment.

Verify:

```powershell
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY
```

Expected:

- `torch.cuda.is_available()` prints `True`.
- GPU has 8GB VRAM.

- [ ] **Step 2: Run formal 512 training on Windows**

Run:

```powershell
python -m model.train_gan `
  --image-size 512 `
  --processed-dir dataset/generate_data/processed_512_front `
  --output-dir outputs/W512_001_g96_d32 `
  --epochs 3000 `
  --batch-size 4 `
  --generator-features 96 `
  --discriminator-features 32 `
  --discriminator-norm none `
  --discriminator-mode global `
  --adversarial-loss-mode bce `
  --d-lr 0.0001 `
  --g-lr 0.0002 `
  --diffaugment `
  --diffaugment-policy color,translation,cutout `
  --ema-decay 0.999 `
  --target-prior-weight 0.0 `
  --amp `
  --grad-accum-steps 4 `
  --sample-interval 50 `
  --checkpoint-interval 100 `
  --seed 52
```

Expected:

- Training starts on CUDA.
- No out-of-memory error in the first 10 minutes.
- `metrics.json` is written at completion.
- Sample previews exist at epoch 1, every 50 epochs, and final epoch.
- `training_state_latest.pt` is refreshed every 100 epochs and `training_state.pt` is written at completion.

- [ ] **Step 3: Evaluate formal checkpoint with D-score and scorer**

Generate 64 samples:

```powershell
$runDir = Get-ChildItem outputs/W512_001_g96_d32 -Directory -Filter "gan_*" | Sort-Object Name | Select-Object -Last 1
python -m model.generate_samples `
  --checkpoint "$($runDir.FullName)\best_generator.pt" `
  --num 64 `
  --out outputs/W512_001_g96_d32/generated_best `
  --seed 52 `
  --batch-size 4
```

Run D-score:

```powershell
$runDir = Get-ChildItem outputs/W512_001_g96_d32 -Directory -Filter "gan_*" | Sort-Object Name | Select-Object -Last 1
python -m model.evaluate_discriminator `
  --discriminator-checkpoint "$($runDir.FullName)\discriminator.pt" `
  --metrics "$($runDir.FullName)\metrics.json" `
  --real-dir dataset/generate_data/processed_512_front `
  --generated-dir outputs/W512_001_g96_d32/generated_best/samples `
  --generator-checkpoint "$($runDir.FullName)\best_generator.pt" `
  --num-fresh 64 `
  --out outputs/W512_001_g96_d32/discriminator_eval_best.json `
  --batch-size 4
```

Run target scorer:

```powershell
python -m model.score_target `
  --image-dir outputs/W512_001_g96_d32/generated_best/samples `
  --real-dir dataset/generate_data/processed_512_front `
  --out outputs/W512_001_g96_d32/score_best.csv
```

Expected:

- `discriminator_eval_best.json` reports generated `mean_p_real`, `median_p_real`, and `max_p_real`.
- `score_best.csv` is written.
- Final judgment uses both D-score and target scorer, not target scorer alone.

## Decision Rules

Use these thresholds after the Windows formal run:

| Metric | Success band | Interpretation |
|---|---:|---|
| Fresh generated mean `P(real)` | `>= 0.10` | G is beginning to fool D |
| Fresh generated median `P(real)` | `>= 0.05` | Improvement is not only from a few lucky samples |
| Fresh generated max `P(real)` | `>= 0.30` | At least some generated samples are close to D-real |
| Target scorer pass rate | `>= 50%` | Target structure is present |
| Diversity ratio | `>= 0.30` | Avoid template collapse |
| Visual inspection | required | Check ring sharpness, paper texture, support structure, background variation |

If mean `P(real) < 0.05` and visual output is still overly smooth, the next architecture change is `residual_upsample` generator, not a larger discriminator.

If target scorer is high but diversity is low, reduce D pressure or add data before increasing G again.

If D-score improves but target rings degrade, add a low-weight ring prior only after preserving photo realism.

## Configuration Table

| Field | Local 512 Smoke | Windows 512 Formal | Rationale |
|---|---:|---:|---|
| `image_size` | `512` | `512` | Better detail than 256 while staying feasible on 8GB VRAM |
| `processed_dir` | `dataset/generate_data/processed_512_front` | `dataset/generate_data/processed_512_front` | Separate 512 ROI cache from current 256 cache |
| `output_dir` | `outputs/smoke_512_g96_d32` | `outputs/W512_001_g96_d32` | Keep smoke and formal outputs separate |
| `epochs` | `1` | `3000` | Smoke checks execution; Windows run trains seriously |
| `max_steps` | `1` | unset | Smoke should not become a real experiment |
| `batch_size` | `1` | `4` | Local CPU smoke is conservative; Windows 8GB should try batch 4 with AMP |
| `grad_accum_steps` | `1` | `4` | Formal effective batch is 16 |
| effective batch | `1` | `16` | Keeps formal training more stable than batch 4 alone |
| `amp` | `false` | `true` | CUDA AMP is required for Windows 8GB VRAM |
| `generator_mode` | `upsample` | `upsample` | Existing stable generator path |
| `generator_features` | `96` | `96` | Strengthens G from current G64 without jumping to G128 |
| `discriminator_mode` | `global` | `global` | Whole ROI image discrimination |
| `discriminator_features` | `32` | `32` | Matches current best D capacity class |
| `discriminator_norm` | `none` | `none` | Current small-data setup avoids D batch norm |
| `adversarial_loss_mode` | `bce` | `bce` | Current successful path uses BCEWithLogits |
| `d_lr` | `0.0001` | `0.0001` | Keep D pressure controlled |
| `g_lr` | `0.0002` | `0.0002` | Preserve current G-favored learning-rate ratio |
| `real_label` | `0.9` | `0.9` | Existing label smoothing |
| `diffaugment` | `true` | `true` | Needed for small data |
| `diffaugment_policy` | `color,translation,cutout` | `color,translation,cutout` | Existing policy |
| `ema_decay` | `0.999` | `0.999` | Stable sampled outputs |
| `target_prior_weight` | `0.0` | `0.0` | First 512 run should test pure GAN photo realism before adding priors |
| `sample_interval` | `1` | `50` | Smoke needs one preview; formal avoids output bloat while showing long-run evolution |
| `checkpoint_interval` | `1` | `100` | Smoke writes final artifacts; formal refreshes rolling recovery state |
| `skip_prepare` | `true` | `false` | Local smoke uses existing prepared 512 cache; Windows formal rebuilds or validates 512 ROI data |
| `min_target_anchor_fraction` | `0.01` | `0.01` | Current ROI acceptance threshold |
| `target_crop_expansion` | `2.9` | `2.9` | Preserve current whole-target ROI framing |
| `seed` | `52` | `52` | Fixed comparison seed |

## Final Check Before Starting Formal Windows Training

Before the Windows formal run, confirm these files exist:

- at least one prepared image exists under `dataset/generate_data/processed_512_front/`
- `configs/gan_512_win_formal.yaml`
- `model/evaluate_discriminator.py`
- latest `outputs/smoke_512_g96_d32/gan_*/metrics.json`
- `outputs/smoke_512_g96_d32/generated_check/contact_sheet.png`
- `outputs/smoke_512_g96_d32/discriminator_eval_smoke.json`

Only after those exist should the Windows formal run start.
