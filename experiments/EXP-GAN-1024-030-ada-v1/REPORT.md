# EXP-GAN-1024-030-ada-v1

Status: dataset and ADA pipeline validated; formal 3000-epoch training not started.

Purpose: isolate the effect of paper-default StyleGAN2-ADA augmentation while keeping the compact
G48/D16 architecture and the learning rates from EXP-GAN-1024-005 unchanged.

- Dataset: immutable real-only `dataset/v1_1024/train`
- Augmentation: online ADA, `bgc`, target 0.6, interval 4, speed 500 kimg, p starts at 0
- Offline random variants: none
- Real-only loader flip: disabled
- Test split: never loaded during training
- Comparison control: EXP-GAN-1024-005-g5e4-long500

Validation completed on 2026-07-17:

- Dataset audit passed: 100 unique accepted sources, 80 train, 0 val, 20 same-scene holdout,
  20 capture groups, no group crossing splits, all hashes and 1024 RGB resolutions verified. Nine
  sources failed the target gate; this includes `IMG_4090.HEIC`, the manually identified pool scene.
- End-to-end 1024 smoke on the real train split passed on CPU with reduced G/D channels; online
  BGC, backward passes, ADA history, metrics, and checkpoint state were all written successfully.
- Full G48/D16 single-batch 1024 forward/backward passed with finite G and D gradients.
- The smoke used a non-formal initial p=0.2 to force execution of the augmentation kernels. The
  formal config remains paper-default p=0 and must adapt from the discriminator statistic.
- This is engineering evidence only, not evidence that the generator now produces arrows.

Review checkpoints: inspect fixed samples and `ada_history.csv` at epochs 100 and 300 before
authorizing the remainder of the 3000-epoch run. Augmentation is successful only if D/G separation
is delayed and arrow/detail structure improves; target rings alone are insufficient.
