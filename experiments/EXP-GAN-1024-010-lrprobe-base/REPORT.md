# EXP-GAN-1024-010-lrprobe-base

- Status: completed
- Evidence level: four-step local learning-rate probe only
- Seed: 42
- Architecture: 1024 px, G48, D16
- Learning rates: G 2e-4, D 1e-4
- EMA: disabled so the four-step raw-generator change remains visible

## Results

- Runtime: 32.28 s on CPU
- Completed steps: 4
- Loss D/G: 0.693015 / 0.693039
- D(real)/D(fake): 0.500284 / 0.500095
- Raw output mean/std: 0.001599 / 0.009363
- Spatial std / sample diversity: 0.000441 / 0.0000217
- Saturated-pixel fraction: 0.0

Finite and stable at four steps. Output remains essentially unstructured, so this
is immediate-stability evidence only.

## Discriminator review pipeline

- Real images: 101/101 classified as real
- Generated probe images: 0/16 classified as fake
- Balanced accuracy: 0.50
- Real mean P(real): 0.500880
- Generated mean P(real): 0.500770
- Populated review groups: `real_as_real`, `fake_as_real`
- Empty review groups: `real_as_fake`, `fake_as_fake`

This discriminator is not useful yet. It sits just above the 0.5 threshold for
almost every input, so the apparent 100% real-image accuracy is a threshold artifact.
The review artifacts prove that per-image correct/error reporting works; they are
not evidence of a trained discriminator.
