# EXP-GAN-1024-006-g5e4-aug-v1-100

- Status: prepared; run only after `EXP-GAN-1024-005-g5e4-long500`
- Purpose: measure the effect of the separate offline augmentation dataset
- Control: `EXP-GAN-1024-005-g5e4-long500`
- Dataset: 500 images at 1024 px (100 original hard links and 400 augmented images)
- Augmentation: horizontal flip, mild affine, mild photometric, and mild affine+photometric
- Training: from scratch, 100 epochs, seed 42
- Architecture: 1024 px, G48, D16
- Learning rates: G 5e-4, D 1e-4
- Loss and online augmentation: unchanged from the control
- Image exposures: 500 images x 100 epochs = 50,000, matching 100 images x 500 epochs
- Samples/checkpoints: every 10 epochs

## Comparison rule

Keep architecture, learning rates, loss, DiffAugment, EMA, seed, and total image exposures
fixed. The intended experimental difference is the offline augmentation pool. Do not run this
experiment concurrently with the control on the same MPS device.

## Results

Pending.
