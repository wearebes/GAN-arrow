```bash
cd ~/GAN-arrow
```

```bash
sudo apt update
```

```bash
sudo apt install -y ffmpeg
```

```bash
conda create -n gan python=3.11 -y
```

```bash
conda activate gan
```

```bash
python -m pip install --upgrade pip
```

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

```bash
python -m pip install pillow matplotlib numpy
```

```bash
nvidia-smi
```

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

The current compact path has one architecture only: 1024 px, bilinear-upsample G48, global D16, and BCEWithLogits. Canary arms use seed 42 and live under `experiments/`, with config, environment, dataset manifest, logs, samples, checkpoints, generation, and discriminator review kept in the same version directory. See `experiments/EXPERIMENT_PLAN.md`.

Build the immutable real-only 1024 dataset once. The current 109-source checkout is split by short
contiguous capture groups into 80 train, 0 validation, and 20 same-scene held-out test images after
the preprocessing acceptance gate (100 accepted sources). No random variants are written to disk.

```bash
python -m model.prepare_v1_dataset --source-dir dataset/origin_data --output-dir dataset/v1_1024 --test-count 20 --val-count 0 --capture-group-size 5 --seed 42
```

The literature-backed augmentation arm uses online ADA (`bgc`, target 0.6) and never reads the test
directory:

```bash
python -u -m model.train_gan --config experiments/EXP-GAN-1024-030-ada-v1/config.json
```

```bash
python -u -m model.train_gan --config experiments/EXP-GAN-1024-001-base/config.json
```

```bash
RUN_DIR=experiments/EXP-GAN-1024-001-base
```

```bash
python -m model.generate_samples --checkpoint "$RUN_DIR/best_generator.pt" --num 64 --out "$RUN_DIR/generated_final" --seed 42 --batch-size 1
```

```bash
python -m model.evaluate_discriminator --discriminator-checkpoint "$RUN_DIR/best_discriminator.pt" --metrics "$RUN_DIR/metrics.json" --real-dir dataset/generate_data/processed_1024_front --generated-dir "$RUN_DIR/generated_final/samples" --generator-checkpoint "$RUN_DIR/best_generator.pt" --num-fresh 64 --out "$RUN_DIR/discriminator_eval.json" --review-dir "$RUN_DIR/discriminator_review" --batch-size 1
```

Stop here and inspect the generated contact sheet plus the four discriminator review sheets. Run the medium- or high-learning-rate arm only after this baseline is reviewable.
