# EXP-GAN-1024-021-mps-throughput

- Status: completed
- Purpose: measure sustained 1024 Apple MPS throughput over eight batches
- Seed: 42
- Architecture: 1024 px, G48, D16
- Learning rates: G 2e-4, D 1e-4
- Gradient accumulation: 8
- Evidence level: device throughput only

## Results

- Device: Apple MPS
- Completed optimizer steps: 8
- Final loss D/G: 0.693128 / 0.693182
- Final D(real)/D(fake): 0.499999 / 0.499980
- Judgment: the short device path completed without NaN or saturation. This is
  throughput/pipeline evidence only and has no model-quality meaning.

After the 2026-07-17 consolidation, metadata and visual diagnostic evidence are
retained; the eight-step model and optimizer checkpoints were removed.
