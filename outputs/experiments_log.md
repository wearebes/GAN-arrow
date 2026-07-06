# GAN Experiments Log

## 2026-07-05 L256 capacity-balance campaign gate

**Execution contract:** This campaign follows the pasted L256-003 / L256-004 / adaptive-extension plan. Training and MPS device checks are executed with the non-sandbox path because the sandbox can report MPS as unavailable on this machine. Probe outputs go under `tmp/`; formal outputs use `outputs/L256_003_g64_d32`, `outputs/L256_004_g64_d64`, and adaptive `outputs/L256_005_*` directories.

**Dataset gate:** `prepared_image_paths(dataset/generate_data/processed_256_front)` reads 61 images. A raw filesystem count sees 62 files, but project training and scoring use the 61-image prepared-path filter.

**Device gate command output:**

```text
/opt/anaconda3/envs/gan/lib/python3.13/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
arm64 2.6.0 True
```

**3-epoch probe:** `tmp/probe_device_gate/gan_20260705_213814`

**Probe command output excerpt:**

```text
epoch=001 loss_d=0.6893 loss_g=0.6887 d_real=0.5074 d_fake=0.5019
epoch=002 loss_d=0.6903 loss_g=0.6665 d_real=0.5375 d_fake=0.5247
epoch=003 loss_d=0.6851 loss_g=0.7071 d_real=0.5166 d_fake=0.5048
"prepared_count": 61
"completed_steps": 12
"device": "mps"
"generator": 2482112
"discriminator": 2790912
```

**Probe result:** Passed. Device was `mps`; elapsed wall time was about 81 seconds, below the 2-minute gate.

## L256-003 - capacity_balance_g64_d32_1000e

**Hypothesis:** D64 is likely over-capacity for the 61-image 256px ROI set, learning the real/fake boundary fast enough to suppress G. Reducing D to features=32 makes D parameter count comparable to G (G 2,482,112; D 2,790,912), so within 4000 updates the run should enter the planned balance band (`d_fake` 0.15-0.45) and begin forming concentric target structure.

**Config:**
- image_size: 256
- processed_dir: `dataset/generate_data/processed_256_front`
- generator: upsample, features=64
- discriminator: global, features=32, norm=none
- adversarial_loss_mode: bce
- batch_size: 16
- epochs: 4 segments x 250 = 1000
- d_lr: 0.0001
- g_lr: 0.0002
- diffaugment: color,translation,cutout
- ema_decay: 0.999
- seeds: 42, 43, 44, 45
- output_dir: `outputs/L256_003_g64_d32`

**Segment 1 status:** completed.
- formal run_dir: `outputs/L256_003_g64_d32/gan_20260705_214629`
- log: `outputs/L256_003_seg1.log`
- startup-log gate: passed; `epoch=001` and later lines appeared within 5 minutes.
- completed_epochs: 250
- completed_steps: 1000
- device: `mps`
- final loss_d / loss_g: 0.5051 / 1.5438
- final d_real / d_fake: 0.6145 / 0.2663
- tail20 loss_d / loss_g: 0.4938 / 1.5521
- tail20 d_real / d_fake: 0.6723 / 0.3039
- score: 0 / 64 = 0.00%
- diversity ratio: 0.3059
- score files: `outputs/L256_003_g64_d32/score_seg1.csv`, `outputs/L256_003_g64_d32/score_seg1_summary.json`
- generated samples: `outputs/L256_003_g64_d32/generated_seg1/samples`
- visual note: blurred green/gray paper-like blocks and soft shadows; no visible red/yellow/blue concentric ring structure.

**Segment 1 conclusion:** Device gate passed and diversity barely passed, but target-geometry score remains 0/64. Tail metrics do not meet the allowed D-too-strong or D-too-weak intervention rules, so L256-003 continues to segment 2 with unchanged hyperparameters and full `training_state.pt` resume.

**Segment 2 status:** completed.
- formal run_dir: `outputs/L256_003_g64_d32/gan_20260705_233538`
- resume_training_state: `outputs/L256_003_g64_d32/gan_20260705_214629/training_state.pt`
- seed: 43
- log: `outputs/L256_003_seg2.log`
- startup-log gate: passed; `epoch=001` appeared within 5 minutes.
- completed_epochs_this_segment: 250
- cumulative_epochs: 500
- completed_steps_this_segment: 1000
- device: `mps`
- final loss_d / loss_g: 0.5524 / 1.7685
- final d_real / d_fake: 0.6989 / 0.3001
- tail20 loss_d / loss_g: 0.4382 / 1.9534
- tail20 d_real / d_fake: 0.7305 / 0.2493
- score: 31 / 64 = 48.44%
- diversity ratio: 0.5106
- score files: `outputs/L256_003_g64_d32/score_seg2.csv`, `outputs/L256_003_g64_d32/score_seg2_summary.json`
- generated samples: `outputs/L256_003_g64_d32/generated_seg2/samples`
- visual note: clear but blurred target-paper structure with blue outer ring and red center; some targets are off-center, elliptical, or weakly separated.

**500-epoch stop-loss check:** Not triggered. Pass rate is not 0, diversity is above 0.30, and samples visibly contain ring-like target structure. Continue L256-003 to segment 3 with unchanged hyperparameters.

**Segment 2 conclusion:** D32 produced a strong improvement from segment 1, reaching 48.44% pass rate but still below the success exit threshold of 50%. Tail metrics remain within the no-intervention region, so continue segment 3 from `training_state.pt`.

**Segment 3 status:** completed.
- formal run_dir: `outputs/L256_003_g64_d32/gan_20260706_005626`
- resume_training_state: `outputs/L256_003_g64_d32/gan_20260705_233538/training_state.pt`
- seed: 44
- log: `outputs/L256_003_seg3.log`
- startup-log gate: passed; `epoch=001` appeared within 5 minutes.
- completed_epochs_this_segment: 250
- cumulative_epochs: 750
- completed_steps_this_segment: 1000
- device: `mps`
- final loss_d / loss_g: 0.5415 / 1.8909
- final d_real / d_fake: 0.6990 / 0.3260
- tail20 loss_d / loss_g: 0.4304 / 2.0607
- tail20 d_real / d_fake: 0.7250 / 0.2408
- score: 39 / 64 = 60.94%
- diversity ratio: 0.5907
- score files: `outputs/L256_003_g64_d32/score_seg3.csv`, `outputs/L256_003_g64_d32/score_seg3_summary.json`
- generated samples: `outputs/L256_003_g64_d32/generated_seg3/samples`
- visual note: clear target-paper structure with yellow center, red ring, blue outer ring, and white backing; still soft/blurred with mild elliptical distortion.

**Segment 3 conclusion:** This segment crosses the success threshold (pass rate >=50%) while preserving diversity. It is the current best L256-003 checkpoint, but the campaign still continues to L256-003 segment 4 and L256-004 because the plan requires complete dual-arm evidence before attribution.

**Segment 4 status:** completed.
- formal run_dir: `outputs/L256_003_g64_d32/gan_20260706_022859`
- resume_training_state: `outputs/L256_003_g64_d32/gan_20260706_005626/training_state.pt`
- seed: 45
- log: `outputs/L256_003_seg4.log`
- startup-log gate: passed; `epoch=001` appeared within 5 minutes.
- completed_epochs_this_segment: 250
- cumulative_epochs: 1000
- completed_steps_this_segment: 1000
- device: `mps`
- final loss_d / loss_g: 0.3626 / 2.2829
- final d_real / d_fake: 0.7863 / 0.1994
- tail20 loss_d / loss_g: 0.4044 / 2.0768
- tail20 d_real / d_fake: 0.7343 / 0.2236
- score: 64 / 64 = 100.00%
- diversity ratio: 0.4952
- score files: `outputs/L256_003_g64_d32/score_seg4.csv`, `outputs/L256_003_g64_d32/score_seg4_summary.json`
- generated samples: `outputs/L256_003_g64_d32/generated_seg4/samples`
- visual note: stable target-paper layout with clear yellow center, red ring, blue ring, gray/white backing, and moderate smoothing; all samples remain stylized GAN outputs but visually target-like.

**L256-003 arm conclusion:** G64+D32 completed all 1000 epochs on MPS. The arm improved from 0.00% (seg1) to 48.44% (seg2), 60.94% (seg3), and 100.00% (seg4), with diversity always at or above 0.30 after seg1. No decision-table intervention was legally triggered. Current best checkpoint is `outputs/L256_003_g64_d32/gan_20260706_022859/best_generator.pt`.

## L256-004 - capacity_balance_g64_d64_1000e

**Hypothesis:** With the same budget, G64+D64 should show whether the larger discriminator was the main source of the earlier 256px failure. If D64 overpowers G, it should more quickly enter higher `d_real` / lower `d_fake` dynamics and should not outperform L256-003 under the same seeds and 1000-epoch budget.

**Config:**
- image_size: 256
- processed_dir: `dataset/generate_data/processed_256_front`
- generator: upsample, features=64
- discriminator: global, features=64, norm=none
- adversarial_loss_mode: bce
- batch_size: 16
- epochs: 4 segments x 250 = 1000
- d_lr: 0.0001
- g_lr: 0.0002
- diffaugment: color,translation,cutout
- ema_decay: 0.999
- seeds: 42, 43, 44, 45
- output_dir: `outputs/L256_004_g64_d64`
- intervention policy: no decision-table intervention; this is a raw dynamics control arm.

**Segment 1 status:** completed.
- formal run_dir: `outputs/L256_004_g64_d64/gan_20260706_035751`
- seed: 42
- log: `outputs/L256_004_seg1.log`
- startup-log gate: passed; `epoch=001` appeared within 5 minutes.
- completed_epochs_this_segment: 250
- cumulative_epochs: 250
- completed_steps_this_segment: 1000
- device: `mps`
- final loss_d / loss_g: 0.3303 / 2.5796
- final d_real / d_fake: 0.7343 / 0.1143
- tail20 loss_d / loss_g: 0.3697 / 2.5710
- tail20 d_real / d_fake: 0.7852 / 0.1959
- score: 0 / 64 = 0.00%
- diversity ratio: 0.2965
- score files: `outputs/L256_004_g64_d64/score_seg1.csv`, `outputs/L256_004_g64_d64/score_seg1_summary.json`
- generated samples: `outputs/L256_004_g64_d64/generated_seg1/samples`
- visual note: pale paper-like blocks with blue-gray blurry patches; no red/yellow target-ring structure.

**Segment 1 conclusion:** D64 shows a much stronger early discriminator regime than L256-003, including repeated low `d_fake` and high `loss_g`, but generated samples do not yet contain target structure and diversity is just below the 0.30 gate. Continue the raw D64 control arm without intervention.

**Segment 2 status:** completed.
- formal run_dir: `outputs/L256_004_g64_d64/gan_20260706_054224`
- resume_training_state: `outputs/L256_004_g64_d64/gan_20260706_035751/training_state.pt`
- seed: 43
- log: `outputs/L256_004_seg2.log`
- startup-log gate: passed; `epoch=001` appeared within 5 minutes.
- completed_epochs_this_segment: 250
- cumulative_epochs: 500
- completed_steps_this_segment: 1000
- device: `mps`
- final loss_d / loss_g: 0.2206 / 3.8611
- final d_real / d_fake: 0.8649 / 0.0516
- tail20 loss_d / loss_g: 0.3181 / 2.9704
- tail20 d_real / d_fake: 0.8047 / 0.1449
- score: 60 / 64 = 93.75%
- diversity ratio: 0.2066
- score files: `outputs/L256_004_g64_d64/score_seg2.csv`, `outputs/L256_004_g64_d64/score_seg2_summary.json`
- generated samples: `outputs/L256_004_g64_d64/generated_seg2/samples`
- visual note: target-paper structure is present, but outputs are heavily templated and smoothed; low diversity gate failure is visually consistent with the score.

**Segment 2 conclusion:** D64 can recover target-like structure by 500 cumulative epochs, but it does so with strong discriminator pressure and poor sample diversity. This is not a clean improvement over L256-003 because the pass rate is high while diversity fails; continue the raw D64 control arm without intervention.

**Segment 3 status:** completed.
- formal run_dir: `outputs/L256_004_g64_d64/gan_20260706_072559`
- resume_training_state: `outputs/L256_004_g64_d64/gan_20260706_054224/training_state.pt`
- seed: 44
- log: `outputs/L256_004_seg3.log`
- startup-log gate: passed; `epoch=001` appeared within 5 minutes.
- completed_epochs_this_segment: 250
- cumulative_epochs: 750
- completed_steps_this_segment: 1000
- device: `mps`
- final loss_d / loss_g: 0.3495 / 3.4105
- final d_real / d_fake: 0.8053 / 0.1559
- tail20 loss_d / loss_g: 0.3394 / 2.7038
- tail20 d_real / d_fake: 0.8068 / 0.1611
- score: 64 / 64 = 100.00%
- diversity ratio: 0.2190
- score files: `outputs/L256_004_g64_d64/score_seg3.csv`, `outputs/L256_004_g64_d64/score_seg3_summary.json`
- generated samples: `outputs/L256_004_g64_d64/generated_seg3/samples`
- visual note: target-paper geometry is clear and consistent, but the grid is highly templated with little pose/background variation.

**Segment 3 conclusion:** D64 reaches perfect target-pass scoring by 750 cumulative epochs, but diversity remains far below the gate. Relative to L256-003, the larger discriminator does not produce a better deliverable tradeoff because it converges to a sharper but more collapsed template.

**Segment 4 status:** completed.
- formal run_dir: `outputs/L256_004_g64_d64/gan_20260706_090619`
- resume_training_state: `outputs/L256_004_g64_d64/gan_20260706_072559/training_state.pt`
- seed: 45
- log: `outputs/L256_004_seg4.log`
- startup-log gate: passed; `epoch=001` appeared within 5 minutes.
- completed_epochs_this_segment: 250
- cumulative_epochs: 1000
- completed_steps_this_segment: 1000
- device: `mps`
- final loss_d / loss_g: 0.2823 / 3.0603
- final d_real / d_fake: 0.8203 / 0.1090
- tail20 loss_d / loss_g: 0.3191 / 2.7844
- tail20 d_real / d_fake: 0.8218 / 0.1375
- score: 64 / 64 = 100.00%
- diversity ratio: 0.2094
- score files: `outputs/L256_004_g64_d64/score_seg4.csv`, `outputs/L256_004_g64_d64/score_seg4_summary.json`
- generated samples: `outputs/L256_004_g64_d64/generated_seg4/samples`
- visual note: clean and centered target-paper template, but sample-to-sample variation is extremely low; diversity failure is visually obvious.

**L256-004 arm conclusion:** G64+D64 completed all 1000 epochs on MPS. It recovered target structure after segment 1 and reached 100% pass rate in segments 3 and 4, but every successful segment failed the diversity gate. The larger discriminator produces sharper template-like targets, not a better deliverable distribution.

## L256 final dual-arm comparison

| Arm | G/D features | Best segment | Best pass rate | Best diversity ratio | Passes diversity gate | Best checkpoint | Judgment |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| L256-003 | G64 / D32 | 4 | 64 / 64 = 100.00% | 0.4952 | yes | `outputs/L256_003_g64_d32/gan_20260706_022859/best_generator.pt` | best deliverable tradeoff |
| L256-004 | G64 / D64 | 3 or 4 | 64 / 64 = 100.00% | 0.2190 (seg3), 0.2094 (seg4) | no | `outputs/L256_004_g64_d64/gan_20260706_090619/best_generator.pt` | sharper but collapsed template |

**Attribution conclusion:** The 256px failure mode was primarily discriminator-capacity imbalance, not insufficient generator capacity. With the same G64 generator and 1000-epoch budget, D32 preserved enough generator variation to reach 100% pass rate with diversity ratio 0.4952, while D64 repeatedly drove low `d_fake` / high `loss_g` windows and converged to low-diversity target templates. The adaptive L256-005 branch is not run because the success-exit condition is already satisfied by L256-003 and the required D64 control arm has been completed.

**Discarded launch attempt:** `outputs/L256_003_g64_d32/gan_20260705_214130` was stopped after 10 epochs because the `tee` pipeline buffered Python stdout and did not produce real-time epoch lines within the 5-minute startup-log gate. It has no `metrics.json` and no checkpoint, is not counted as a formal segment, and must not be used for resume or scoring. Relaunch uses `python -u` only to force unbuffered logging; training hyperparameters are unchanged.

## L256-001 - large256_global_bce_g64d64_diffaug_ema_e300

**Hypothesis:** The previous 256 GAN used only about 0.48M parameters with PatchD and no DiffAugment/EMA, so it could not even show reliable memorization. A 13.63M-parameter 256px global-Discriminator setup should give the discriminator enough global structure capacity and the generator enough image capacity. DiffAugment should reduce small-data discriminator overfit, and EMA should stabilize sampled outputs.

**Config:**
- image_size: 256
- processed_dir: `dataset/generate_data/processed_256_front`
- generator: upsample, features=64
- discriminator: global, features=64, norm=none
- adversarial_loss_mode: bce
- batch_size: 16
- epochs: 300
- d_lr: 0.0002
- g_lr: 0.0002
- real_label: 0.9
- diffaugment: color,translation,cutout
- ema_decay: 0.999
- estimated trainable parameters: G 2,482,112 + D 11,152,384 = 13,634,496

**Training metrics:** pending.

**Scorer pass rate:** pending.

**Diversity metric:** pending.

**Conclusion:** pending.

**Foreground segment:** `outputs/large_256_foreground_chunks/gan_20260705_154722`
- resumed from L256-007 segment 2 `training_state.pt`
- completed_steps in this segment: 12
- final loss_d: 0.3179
- final loss_g: 13.3499
- final d_real: 0.7566
- final d_fake: 0.1565
- target_prior_weight: 10.0 with balanced per-ring loss
- generated_64 pass rate: 41 / 64 = 64.06%
- diversity ratio: 0.5078
- visual note: low-quality but recognizable target-paper images with white paper, blue field, and yellow/orange center

**Conclusion:** This is the first branch that satisfies both the scorer pass gate and diversity gate while also passing visual inspection as low-quality target-like images. It was packaged as `outputs/deliverable_gan/`.

**Foreground segment 1:** `outputs/large_256_foreground_chunks/gan_20260705_151013`
- resumed from L256-005 `training_state.pt`
- target_prior_weight: 50.0
- completed_steps: 12
- final loss_d: 0.3387
- final loss_g: 7.2947
- final d_real: 0.7369
- final d_fake: 0.1648
- generated_64 pass rate: 20 / 64 = 31.25%
- diversity ratio: 0.4391
- visual note: visible white paper, large blue region, and orange center begin to appear

**Foreground segment 2:** `outputs/large_256_foreground_chunks/gan_20260705_152143`
- resumed from segment 1 `training_state.pt`
- target_prior_weight: 50.0
- completed_steps: 12
- final loss_d: 0.2918
- final loss_g: 6.8666
- final d_real: 0.7386
- final d_fake: 0.1040
- generated_64 pass rate: 3 / 64 = 4.69%
- diversity ratio: 0.4409
- visual note: very recognizable low-quality target-paper layout with blue field and orange center, but not enough red/yellow ring separation; score low because blue dominates

**Conclusion:** Strong prior moved visual output closest to a usable target so far, but the original prior was area-dominated by blue/background. I changed `target_ring_prior_loss` to compute balanced per-ring losses so yellow/red are not overwhelmed.

## L256-008 - balanced_ring_prior_from_L256007

**Hypothesis:** The target prior created blue target-paper structure but failed to separate center colors. Balanced per-ring prior loss should strengthen the yellow/red center while preserving the learned white paper and blue field.

**Config delta from L256-007 segment 2:**
- resume_training_state: `outputs/large_256_foreground_chunks/gan_20260705_152143/training_state.pt`
- target_prior_weight: 10.0
- d_lr: 0.00001
- g_lr: 0.0002
- ema_decay: 0.9

**Training metrics:** pending.

**Scorer pass rate:** pending.

**Diversity metric:** pending.

**Conclusion:** pending.

**Foreground segment:** `outputs/large_256_foreground_chunks/gan_20260705_145816`
- resumed from L256-005 `training_state.pt`
- completed_steps in this segment: 12
- final loss_d: 0.4121
- final loss_g: 1.5102
- final d_real: 0.7197
- final d_fake: 0.2632
- target_prior_weight: 5.0
- generated_64 pass rate: 2 / 64 = 3.125%
- diversity ratio: 0.4860
- visual note: white target-paper layout remains, only weak orange/blue blobs; no concentric rings

**Conclusion:** Prior weight 5.0 is too weak to overcome the current image manifold. Try a strong-prior branch from the same L256-005 checkpoint before abandoning this route.

## L256-007 - strong_colored_ring_prior_from_L256005

**Hypothesis:** A weak ring prior did not create visible rings. Increasing `target_prior_weight` to 50 from the white-target-paper checkpoint should test whether the generator can be pulled into explicit concentric target colors while the discriminator keeps the image from becoming purely procedural.

**Config delta from L256-006:**
- resume_training_state: `outputs/large_256_foreground_chunks/gan_20260705_144219/training_state.pt`
- target_prior_weight: 50.0
- d_lr: 0.00001
- g_lr: 0.0002
- ema_decay: 0.9

**Training metrics:** pending.

**Scorer pass rate:** pending.

**Diversity metric:** pending.

**Conclusion:** pending.

**Foreground segment:** `outputs/large_256_foreground_chunks/gan_20260705_144219`
- resumed from L256-004 `training_state.pt`
- effective branch epochs: +3
- completed_steps in this segment: 12
- final loss_d: 0.5188
- final loss_g: 0.4677
- final d_real: 0.8039
- final d_fake: 0.4293
- tail stability judgment: roughly_balanced_short_run
- generated_64 pass rate: 1 / 64 = 1.56%
- diversity ratio: 0.6616
- visual note: white target-paper/board structure is clearer, but colored rings are mostly absent

**Conclusion:** Freezing D and using fast EMA made the image layout more paper-like but lost the target-color signal. The next branch should add an explicit colored-ring prior while keeping GAN training active.

## L256-006 - colored_ring_prior_from_L256005

**Hypothesis:** The model has learned target-paper layout but not the colored concentric rings. Adding a generator-only colored-ring prior should push G toward visible yellow/red/blue target structure, while the discriminator keeps the result close to the real ROI distribution.

**Config delta from L256-005:**
- resume_training_state: `outputs/large_256_foreground_chunks/gan_20260705_144219/training_state.pt`
- d_lr: 0.00001
- g_lr: 0.0002
- ema_decay: 0.9
- target_prior_weight: 5.0
- all other architecture and 256px settings unchanged

**Training metrics:** pending.

**Scorer pass rate:** pending.

**Diversity metric:** pending.

**Conclusion:** pending.

**Foreground segment:** `outputs/large_256_foreground_chunks/gan_20260705_142528`
- branched from the 15-epoch best training state
- effective cumulative epochs on this branch: 18
- completed_steps in this segment: 12
- final loss_d: 0.3297
- final loss_g: 0.9323
- final d_real: 0.8186
- final d_fake: 0.2221
- generated_64 EMA pass rate: 21 / 64 = 32.81%
- EMA diversity ratio: 0.6720
- raw generator pass rate: 51 / 64 = 79.69%
- raw generator diversity ratio: 0.6395
- visual note: raw generator is much closer to target-paper structure but still lacks clear concentric color rings; current scorer is too permissive for this branch

**Conclusion:** The generator itself moved substantially, but EMA with decay 0.999 lagged behind. Machine score alone is insufficient because the raw generator can pass without clean concentric rings. Continue with faster EMA and stronger generator pressure, while keeping visual inspection as a hard gate.

## L256-005 - fixed_d_fast_ema_from_L256004

**Hypothesis:** L256-004 showed raw G improved much more than EMA. Freezing D temporarily (`d_lr=0`) and using faster EMA (`ema_decay=0.9`) may let sampled outputs follow the generator and sharpen target-paper structure without further increasing D pressure.

**Config delta from L256-004:**
- resume_training_state: `outputs/large_256_foreground_chunks/gan_20260705_142528/training_state.pt`
- d_lr: 0.0
- g_lr: 0.0003
- ema_decay: 0.9
- all other architecture and 256px settings unchanged

**Training metrics:** pending.

**Scorer pass rate:** pending.

**Diversity metric:** pending.

**Conclusion:** pending.

**Foreground segment:** `outputs/large_256_foreground_chunks/gan_20260705_141231`
- resumed from L256-002 best `training_state.pt`
- effective cumulative epochs from original start: 18 on this branch
- completed_steps in this segment: 12
- final loss_d: 0.3035
- final loss_g: 1.1762
- final d_real: 0.8061
- final d_fake: 0.1626
- tail stability judgment: generator_struggling
- generated_64 pass rate: 20 / 64 = 31.25%
- diversity ratio: 0.6525
- visual note: more blue/gray, still abstract; pass rate regressed from the 15-epoch checkpoint

**Conclusion:** Reducing D LR to `5e-5` while continuing from 15 epochs did not improve the best score. D remained too strong and pass rate dropped from 35.94% to 31.25%. The next branch should return to the 15-epoch checkpoint and push G harder.

## L256-004 - branch_from_best15_with_generator_heavy_lr

**Hypothesis:** The current best checkpoint is `gan_20260705_135910` at 15 effective epochs with 23/64 pass rate. Since continuing with D LR `5e-5` still led to `generator_struggling`, branch from that best checkpoint with near-frozen D (`d_lr=1e-5`) and stronger G (`g_lr=3e-4`) to see whether target-color geometry can cross the 50% gate.

**Config delta from current best:**
- resume_training_state: `outputs/large_256_foreground_chunks/gan_20260705_135910/training_state.pt`
- d_lr: 0.00001
- g_lr: 0.0003
- all other architecture and 256px settings unchanged

**Training metrics:** pending.

**Scorer pass rate:** pending.

**Diversity metric:** pending.

**Conclusion:** pending.

**Foreground segment:** `outputs/large_256_foreground_chunks/gan_20260705_131829`
- resumed from L256-001 segment 2
- effective cumulative epochs from original start: 9
- completed_steps in this segment: 12
- final loss_d: 0.4970
- final loss_g: 1.1172
- final d_real: 0.7179
- final d_fake: 0.3494
- tail stability judgment: roughly_balanced_short_run
- generated_64 pass rate: 0 / 64 = 0.00%
- diversity ratio: 0.6395
- visual note: improved diversity and stronger colors, still no concentric target geometry

**Interim conclusion:** Lowering D LR improved the balance and diversity metric, but the 256px GAN still has not learned target geometry by 9 effective epochs. Continue only with checkpoint-resumed segments; do not restart from scratch.

**Foreground segment 2:** `outputs/large_256_foreground_chunks/gan_20260705_134459`
- resumed from L256-002 segment 1 G/D/EMA
- effective cumulative epochs from original start: 12
- completed_steps in this segment: 12
- final loss_d: 0.3926
- final loss_g: 1.5274
- final d_real: 0.7783
- final d_fake: 0.2751
- generated_64 pass rate: 2 / 64 = 3.125%
- diversity ratio: 0.6444
- artifact note: first run that saved full `training_state.pt` for optimizer-state resume
- visual note: still color blobs; two scoring passes are weak local-color hits, not deliverable target images

**Foreground segment 3:** `outputs/large_256_foreground_chunks/gan_20260705_135910`
- resumed from segment 2 `training_state.pt`, preserving optimizer state
- effective cumulative epochs from original start: 15
- completed_steps in this segment: 12
- final loss_d: 0.3590
- final loss_g: 1.2884
- final d_real: 0.7939
- final d_fake: 0.2038
- tail stability judgment: generator_struggling
- generated_64 pass rate: 23 / 64 = 35.94%
- diversity ratio: 0.6373
- visual note: much more target-color presence by scorer, but still abstract and not a clean concentric target

**Interim conclusion after 15 epochs:** The 256px 13.63M-param route is now producing measurable target-color signals and passes diversity, but D pressure is high and the visual geometry is still not deliverable. Next segment should reduce D LR to `5e-5` while keeping G LR at `2e-4`.

## L256-003 - continue_from_training_state_with_weaker_d

**Hypothesis:** At 15 effective epochs, pass rate jumped to 35.94% but `d_fake` fell to about 0.20 and the diagnostic is `generator_struggling`. Resuming full optimizer state while reducing `d_lr` to 0.00005 should keep learned signal while giving G more room to form target-like color geometry.

**Config delta from L256-002:**
- resume_training_state: `outputs/large_256_foreground_chunks/gan_20260705_135910/training_state.pt`
- d_lr: 0.00005
- g_lr: 0.0002
- all other architecture and 256px settings unchanged

**Training metrics:** pending.

**Scorer pass rate:** pending.

**Diversity metric:** pending.

**Conclusion:** pending.

**Scorer calibration before long run:**
- real 256 ROI pass rate: 60 / 61 = 98.36%
- prior failed GAN tile pass rate: 3 / 16 = 18.75%
- calibration record: `outputs/scorer_calibration.md`

**Next step:** Run the full 256 large model and score 64 EMA-generated samples from `best_generator.pt`.

**Foreground segment 1:** `outputs/large_256_foreground_chunks/gan_20260705_124916`
- effective cumulative epochs: 3
- completed_steps: 12
- final loss_d: 0.5056
- final loss_g: 1.4403
- final d_real: 0.7486
- final d_fake: 0.4100
- generated_64 pass rate: 0 / 64 = 0.00%
- diversity ratio: 0.2796
- visual note: soft low-frequency color fields, no target structure yet

**Foreground segment 2:** `outputs/large_256_foreground_chunks/gan_20260705_130411`
- resumed from segment 1 G/D/EMA
- effective cumulative epochs: 6
- completed_steps in this segment: 12
- final loss_d: 0.6586
- final loss_g: 1.4094
- final d_real: 0.7626
- final d_fake: 0.3672
- generated_64 pass rate: 0 / 64 = 0.00%
- diversity ratio: 0.6166
- visual note: stronger color diversity, but still abstract magenta/gray blobs rather than concentric target rings

**Interim conclusion:** The 13.63M-parameter 256px model is training and no longer mode-collapsed by the diversity metric, but the generator has not reached target-color geometry. D strengthened quickly, so the next segment should reduce D pressure rather than simply continue at the same LR.

## L256-002 - continue_from_L256001_with_lower_d_lr

**Campaign note:** Superseded by the 2026-07-05 L256-003 capacity-balance campaign above; this legacy short-run branch is no longer an active execution target for the current plan.

**Hypothesis:** After 6 effective epochs, D has started pushing `d_fake` down while G loss remains high. Continuing from the current checkpoint with `d_lr=0.0001` and `g_lr=0.0002` may let G catch up without abandoning the 256px 13.63M-parameter setup.

**Config delta from L256-001:**
- resume_generator: `outputs/large_256_foreground_chunks/gan_20260705_130411/generator.pt`
- resume_discriminator: `outputs/large_256_foreground_chunks/gan_20260705_130411/discriminator.pt`
- resume_ema_generator: `outputs/large_256_foreground_chunks/gan_20260705_130411/generator_ema.pt`
- d_lr: 0.0001
- g_lr: 0.0002
- all other architecture and 256px settings unchanged

**Training metrics:** pending.

**Scorer pass rate:** pending.

**Diversity metric:** pending.

**Conclusion:** pending.
