# EXP-GAN-1024-000-local-smoke

- Status: completed
- Purpose: one real CPU forward/backward step through the exact 1024 px G48/D16 pipeline
- Seed: 42
- Learning rates: G 2e-4, D 1e-4
- Evidence level: pipeline smoke only; not a model-quality result

## Results

- Device: CPU
- Runtime: 25.17 s
- Prepared images: 101/109, all exactly 1024x1024
- Completed optimizer steps: 1
- Parameters: G 1,592,208; D 1,223,424; total 2,815,632
- First-step metrics: loss D 0.6931, loss G 0.6931, D(real) 0.5001, D(fake) 0.5000
- Original artifact check: config, environment, preprocessing report, 101-image
  manifest, log, history, sample, best G/D, final G/D, EMA G, and resumable training
  states were all produced successfully.

The generated sample is near-uniform after one step, as expected from an untrained
model. This experiment proves the 1024 pipeline and artifact contract only; it is not
evidence that the model has learned target or arrow structure.

After the 2026-07-17 consolidation, only metadata and visual pipeline evidence are
retained. The untrained model and optimizer checkpoints were removed.
