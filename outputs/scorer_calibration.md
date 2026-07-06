# Target Scorer Calibration

## Frozen Inputs

- Positive set: `dataset/generate_data/processed_256_front`
- Positive count at calibration time: 61 prepared 256px ROI images
- Negative set: 16 tiles split from `outputs/final_front256_patch_lsgan_g16_d16_e60/gan_20260705_024056/samples_epoch_060.png`

## Frozen Thresholds

- blue_fraction >= 0.003
- inner_fraction >= 0.01
- color_fraction >= 0.025
- color_fraction <= 0.22
- inner_mean_radius < blue_mean_radius * 2.5
- radius_cv <= 0.85

## Calibration Result

- Positive pass rate: 60 / 61 = 98.36%
- Historical failed GAN pass rate: 3 / 16 = 18.75%

## Interpretation

The scorer is calibrated as a 256px ROI-level diagnostic. It rejects blank/gray images, images without enough red/yellow/blue target color, and large abstract color blobs that dominated the prior failed GAN sample. It is not a general detector for full-frame photos or clean procedural targets that fill most of the image.
