# GAN-arrow experiment summary

Updated: 2026-07-17

## Executive conclusion

The repository now contains two distinct experiment lines and they should not be
reported as one result:

1. The earlier 256 px campaign learned recognizable target-face colour and ring
   structure. Its strongest documented capacity-balance arm was G64/D32 at 1000
   cumulative epochs, with a ring-heuristic pass rate of 64/64 and diversity ratio
   0.4952. This is evidence for target-face generation, not evidence that arrow
   shafts or impact points were learned.
2. The current compact 1024 px campaign uses one controlled architecture:
   bilinear-upsample G48, global D16, BCEWithLogits, DiffAugment, EMA, and seed 42.
   Raising resolution and training longer improved the coarse target silhouette,
   but no completed run learned identifiable arrows or impact detail. The best
   1024 px run is therefore research evidence, not a downstream-data generator.

The main 1024 px failure mode is not numerical instability. Training remains finite,
but the discriminator increasingly separates real and generated images while the
generator converges to smooth, low-detail target templates. More epochs with the same
objective are not justified.

## 1024 px experiment matrix

| Experiment | Evidence level | Result | Decision |
|---|---|---|---|
| `EXP-GAN-1024-000-local-smoke` | Pipeline smoke | One CPU step completed; losses near 0.693; output unstructured | Pipeline contract only |
| `EXP-GAN-1024-010/011/012` | Four-step LR probes | All finite; high-LR arm moved fastest but did not saturate | Stability screen only; no quality claim |
| `EXP-GAN-1024-001-base` | Formal, 20 epochs | Early-stopped; final D/G loss 0.5216/1.4225; no recognizable target or arrow | Too short and already D-led |
| `EXP-GAN-1024-004-base-long100` | Formal, 100 epochs | Coarse target/stand silhouette; no clear rings or arrows; balanced D accuracy 99.50% | Longer training alone fails |
| `EXP-GAN-1024-005-g5e4-long500` | Formal, 500 epochs | 64/64 coarse target-like layouts, but blurred rings, no arrows, limited diversity; D balanced accuracy 100% | Best 1024 visual result, still fails the task |
| `EXP-GAN-1024-006-g5e4-aug-v1-100` | Prepared only | Offline-augmentation control has not run | Valid next controlled comparison, not a result |
| `EXP-GAN-1024-021-mps-throughput` | Device diagnostic | Eight MPS steps completed | Throughput/pipeline evidence only |
| `EXP-GAN-1024-002/003` | Planned only | No training outputs | Keep as planned arms, not completed experiments |

## Formal-run comparison

| Run | Epochs | Prepared images | Tail loss D/G | Tail D(real)/D(fake) | Visual outcome |
|---|---:|---:|---:|---:|---|
| `001-base` | 20 | 101 | 0.5150 / 1.1453 | 0.7243 / 0.3602 | No recognizable target or arrow |
| `004-base-long100` | 100 | 101 | 0.3563 / 1.9105 | 0.8340 / 0.1787 | Coarse silhouette; no rings/arrows |
| `005-g5e4-long500` | 500 | 100 | 0.2888 / 2.4479 | 0.8543 / 0.1207 | Recognizable target template; no arrow/detail |

The progression is consistent: visual target structure improves with exposure, while
the discriminator margin widens and the generator loss rises. This rules out the
simple claim that the 1024 px model only needed more time.

## Current decision

- Do not continue `005` with unchanged data, architecture, and adversarial loss.
- Do not treat the discriminator's high accuracy as a quality success; here it is
  evidence that generated images remain easy to reject.
- If experimentation continues, run `006` only as the declared exposure-matched
  offline-augmentation comparison. Its result should be judged on visible arrows,
  impact points, ring fidelity, and diversity, not loss alone.
- Keep the earlier 256 px result and the current 1024 px result separate: the former
  establishes target-face learnability; the latter shows that higher resolution does
  not by itself recover arrow detail.

## Artifact retention after cleanup

Formal runs `001`, `004`, and `005` retain configuration, environment, dataset
manifest, logs, history, metrics, loss curve, best/final G, best/final D aliases,
two final resumable states, final generated samples/contact sheet, and discriminator
review evidence. Periodic resumable states were removed after final-state validation.

Retained fixed-noise milestones are:

- `001`: epochs 5 and 20.
- `004`: epoch 100.
- `005`: epochs 20, 30, 300, 350, 380, and 500.

Smoke, four-step LR-probe, and throughput runs retain their reports, configs,
environment, manifests, histories, metrics, logs, loss curves, and fixed-noise visual
evidence. Their model and optimizer checkpoints were removed because those runs are
explicitly non-quality diagnostics and are reproducible from their recorded configs.
