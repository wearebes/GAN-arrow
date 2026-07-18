# 1024 learning-rate probe summary

All arms use the same 101 prepared images, 1024 px G48/D16 architecture, seed 42,
first four shuffled samples, BCE loss, and DiffAugment. EMA is disabled only for
these probes so the raw four-step generator change remains observable.

| Probe | G LR | D LR | Loss D | Loss G | D(real) | D(fake) | Output mean | Output std | Saturated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 2e-4 | 1e-4 | 0.693015 | 0.693039 | 0.500284 | 0.500095 | 0.001599 | 0.009363 | 0.0 |
| mid | 5e-4 | 1e-4 | 0.693022 | 0.693030 | 0.500267 | 0.500088 | -0.000864 | 0.003740 | 0.0 |
| high | 1e-3 | 5e-4 | 0.693674 | 0.683485 | 0.510925 | 0.508641 | -0.021129 | 0.001397 | 0.0 |

The high arm is finite and does not immediately saturate, so 1e-3 is not rejected.
It also changes the adversarial scores and output mean much faster than the other
arms, so four steps do not justify using it as the default. Formal order remains
base, then mid, then high only if the first two are stable but learn too slowly.

These probes contain no learned target or arrow structure and must not be used as
model-quality evidence.
