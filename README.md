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

```bash
python -m model.train_gan --dataset-dir dataset/origin_data --processed-dir dataset/generate_data/processed_512_front --output-dir outputs/W512_001_g96_d32 --image-size 512 --epochs 3000 --batch-size 4 --generator-mode upsample --generator-features 96 --discriminator-mode global --discriminator-features 32 --discriminator-norm none --adversarial-loss-mode bce --d-lr 0.0001 --g-lr 0.0002 --diffaugment --diffaugment-policy color,translation,cutout --ema-decay 0.999 --target-prior-weight 0.0 --amp --grad-accum-steps 4 --sample-interval 50 --checkpoint-interval 100 --min-target-anchor-fraction 0.01 --target-crop-expansion 2.9 --seed 52
```

```bash
RUN_DIR=$(find outputs/W512_001_g96_d32 -maxdepth 1 -type d -name 'gan_*' | sort | tail -n 1)
```

```bash
python -m model.generate_samples --checkpoint "$RUN_DIR/best_generator.pt" --num 64 --out outputs/W512_001_g96_d32/generated_final --seed 52 --batch-size 4
```

```bash
python -m model.evaluate_discriminator --discriminator-checkpoint "$RUN_DIR/discriminator.pt" --metrics "$RUN_DIR/metrics.json" --real-dir dataset/generate_data/processed_512_front --generated-dir outputs/W512_001_g96_d32/generated_final/samples --generator-checkpoint "$RUN_DIR/best_generator.pt" --num-fresh 64 --out outputs/W512_001_g96_d32/discriminator_eval_final.json --batch-size 4
```

```bash
python -m model.train_gan --dataset-dir dataset/origin_data --processed-dir dataset/generate_data/processed_512_front --output-dir outputs/W512_001_g96_d32_resume --image-size 512 --epochs 3000 --batch-size 4 --generator-mode upsample --generator-features 96 --discriminator-mode global --discriminator-features 32 --discriminator-norm none --adversarial-loss-mode bce --d-lr 0.0001 --g-lr 0.0002 --diffaugment --diffaugment-policy color,translation,cutout --ema-decay 0.999 --target-prior-weight 0.0 --amp --grad-accum-steps 4 --sample-interval 50 --checkpoint-interval 100 --min-target-anchor-fraction 0.01 --target-crop-expansion 2.9 --seed 52 --resume-training-state "$RUN_DIR/training_state_latest.pt"
```
