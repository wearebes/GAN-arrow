# EXP-GAN-1024-012-lrprobe-high

- Status: completed
- Evidence level: four-step local learning-rate probe only
- Seed: 42
- Architecture: 1024 px, G48, D16
- Learning rates: G 1e-3, D 5e-4
- EMA: disabled so the four-step raw-generator change remains visible

## Results

- Runtime: 31.18 s on CPU
- Completed steps: 4
- Loss D/G: 0.693674 / 0.683485
- D(real)/D(fake): 0.510925 / 0.508641
- Raw output mean/std: -0.021129 / 0.001397
- Spatial std / sample diversity: 0.000665 / 0.0001991
- Saturated-pixel fraction: 0.0

No NaN or immediate pixel saturation occurred, but this arm moved farther from
initialization than the other two in only four steps. Keep it as the third,
high-risk canary rather than making 1e-3 the default.
