# EXP-GAN-1024-005-g5e4-long500

- Status: completed on Apple MPS
- Purpose: test whether increasing only generator learning rate improves target and arrow detail
- Control: `EXP-GAN-1024-004-base-long100`
- Training: from scratch, maximum 500 epochs, seed 42
- Dataset: 100 prepared 1024 px images; confirmed no-target `IMG_4090` excluded
- Architecture: 1024 px, G48, D16, 2,815,632 trainable parameters
- Learning rates: G 5e-4, D 1e-4
- Loss: BCEWithLogits adversarial loss only; no MSE or auxiliary loss
- Regularization: DiffAugment and EMA 0.995
- Samples: every 10 epochs
- Recoverable checkpoints: every 25 epochs
- Early stopping: disabled so the requested 500-epoch behavior can be observed

## Controlled variables

Relative to the previous formal configuration, the intended optimization change is only
G learning rate from 2e-4 to 5e-4. The one confirmed no-target crop is excluded as a data
correction and recorded explicitly. Architecture, D learning rate, seed, loss, resolution,
augmentation, EMA, batch behavior, and preprocessing crop policy remain unchanged.

## Evaluation gates

- Canary: inspect losses and fixed samples at epochs 10 and 25 for NaN, saturation, or collapse.
- Progress: inspect epochs 50, 100, 200, 300, 400, and 500 using the same fixed-noise panel.
- Final: use the last-five-epoch loss averages plus a 64-image final sample set.
- Discriminator review: export real-as-real, real-as-fake, fake-as-fake, and fake-as-real groups.
- Success requires visible target rings and arrow-like structures; merely producing a coarse
  target silhouette does not pass.

## Results

- Epoch 1: loss D 0.6930, loss G 0.6924, D(real) 0.5008, D(fake) 0.5005.
- Startup judgment: finite and balanced at initialization; no immediate saturation.
- Epoch 20 fixed samples: uniform brown output with no target or arrow structure yet.
- Epoch 30 fixed samples: uniform purple output with no target or arrow structure; the output
  color changed, but diversity and geometry have not emerged.
- Epoch 25: loss D 0.4613, loss G 1.2047, D(real) 0.7824, D(fake) 0.3277.
- Epoch 27: loss D 0.3616, loss G 1.6223, D(real) 0.8206, D(fake) 0.2086.
- Canary judgment: finite, but D is already pulling ahead and the generator has no visible
  structure at epoch 20. Continue to the declared epoch 50 gate before deciding whether this
  is slow emergence or an unproductive high-G-LR trajectory.
- Epoch 300 fixed samples: recognizable target face with black, blue, red, and yellow regions;
  still blurred, no arrow shafts or impact detail.
- Epoch 350 fixed samples: target geometry becomes more regular, while samples become more
  alike, indicating reduced diversity.
- Epoch 380 fixed samples: stable target-like structure remains, but concentric rings are
  simplified and arrows remain absent.
- Epoch 368-387 mean: loss D 0.4064, loss G 1.8118, D(real) 0.7455, D(fake) 0.2103.
- Progress judgment at epoch 387/500: D is stronger but training is finite. The run has learned
  the coarse target distribution; it has not learned the high-frequency arrow/detail objective.
- Epoch 500: loss D 0.2837, loss G 2.5196, D(real) 0.8441, D(fake) 0.1285.
- Last five epochs: mean loss D 0.2888, mean loss G 2.4479, mean D(real) 0.8543,
  mean D(fake) 0.1207. Final loss judgment: generator struggling.
- Final visual review: 64/64 samples have a recognizable coarse target-like arrangement, but
  they are heavily smoothed, have simplified/missing scoring rings, show no identifiable arrow
  shafts or impact points, and have limited structural diversity.
- Final discriminator review: 100/100 cleaned real images classified as real and 64/64 saved
  generated images classified as fake; balanced accuracy 100% at threshold 0.5.
- Mean P(real): real set 0.984425; saved generated set 0.00000106. No generated image fooled D.

## Final decision

- The 500-epoch higher-G-LR run improves coarse target recognition compared with the 100-epoch
  baseline, but does not meet the arrow/detail requirement.
- Further continuation with the same configuration is not justified: D separation is widening
  and G loss rises near the end.
- Retain `final_generator.pt`, `training_state.pt`, and `training_state_latest.pt` as
  the final/recoverable records. Periodic recovery states were removed after final-state
  validation during the 2026-07-17 artifact consolidation. Do not use this generator
  to create downstream ring-scoring training data.
