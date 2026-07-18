# EXP-GAN-1024-004-base-long100

- Status: completed on Apple MPS
- Purpose: test whether shapes emerge with substantially longer training
- Parent: `EXP-GAN-1024-001-base`, resumed from epoch 20
- Target: total epoch 100
- Seed: 42
- Architecture: 1024 px, G48, D16
- Learning rates: G 2e-4, D 1e-4
- Early stopping: disabled for this long-run test
- Samples/checkpoints: every 10 absolute epochs

## Decision rule

Use final loss plus the last-five-epoch averages as the primary stability evidence.
Inspect only the final epoch-100 fixed samples and generated set for target/arrow shape;
do not reject the run based on early mosaic images.

## Results

- Resume verification: epoch numbering continued at 21 and optimizer state loaded.
- Epoch 21: loss D 0.4429, loss G 1.1950, D(real) 0.7625, D(fake) 0.3203.
- Epoch 100: loss D 0.3316, loss G 1.7738, D(real) 0.8317, D(fake) 0.1819.
- Last five epochs: mean loss D 0.3563, mean loss G 1.9105, mean D(real) 0.8340,
  mean D(fake) 0.1787.
- Loss judgment: generator struggling; the discriminator is substantially stronger.
- Visual judgment: longer training replaced mosaics with a consistent coarse target/stand
  silhouette, but the images remain blurry and contain neither clear concentric scoring rings
  nor identifiable arrow shafts.
- Final discriminator review: 100/101 real images classified as real and 64/64 generated
  images classified as fake; balanced accuracy 99.50% at threshold 0.5.
- The only real-image error is `IMG_4090.png` with P(real)=0.0368. It is primarily an
  entrance/interior scene without a visible target, so it also identifies a dataset-filtering error.
- No generated image was misclassified as real. Mean P(real) for the saved generated set is 0.1306.

## Checkpoint decision

- `final_generator.pt` is the selected deliverable for this experiment and records epoch 100.
- `best_generator.pt` is the same final EMA state because this long-run experiment deliberately
  disabled early stopping and did not define a trustworthy validation-quality metric. It must not
  be described as independently proven “best.”
- The final recoverable aliases `training_state.pt` and `training_state_latest.pt`
  are retained. Periodic recovery states were removed after final-state validation
  during the 2026-07-17 artifact consolidation.
- Under the experiment's declared final-loss rule, epoch 100 is retained, but the result does not
  pass the downstream-data quality requirement.

## Next decision

- Do not increase the discriminator learning rate to 5e-4 or 1e-3: the final discriminator is
  already too strong.
- The next controlled run should change only generator learning rate from 2e-4 to 5e-4, retain
  D=1e-4 and all other settings, and first remove no-target crops such as `IMG_4090`.
