# EXP-GAN-1024-011-lrprobe-mid

- Status: completed
- Evidence level: four-step local learning-rate probe only
- Seed: 42
- Architecture: 1024 px, G48, D16
- Learning rates: G 5e-4, D 1e-4
- EMA: disabled so the four-step raw-generator change remains visible

## Results

- Runtime: 31.72 s on CPU
- Completed steps: 4
- Loss D/G: 0.693022 / 0.693030
- D(real)/D(fake): 0.500267 / 0.500088
- Raw output mean/std: -0.000864 / 0.003740
- Spatial std / sample diversity: 0.000414 / 0.0000299
- Saturated-pixel fraction: 0.0

Finite and stable at four steps, and numerically almost indistinguishable from
the conservative baseline. Longer training is required to judge learning speed.
