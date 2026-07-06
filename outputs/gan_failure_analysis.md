# GAN Failure Analysis

## Current Verdict

The GAN route did not produce a deliverable target-image generator. The best GAN checkpoint is:

- `outputs/final_front256_patch_lsgan_g16_d16_e60/gan_20260705_024056/samples_epoch_060.png`
- Status: `best_256_gan_but_abstract_not_production`

It learned broad target-like color regions, but not a stable concentric target. It is not suitable as a product output.

The currently usable deliverables are:

- `outputs/effective_procedural_targets_256/`
- `outputs/effective_augmented_256/`

## Data Facts

- Raw source images: 109 HEIC files.
- Usable frontal ROI after automatic HEIC stream selection: 53 images.
- Usable tighter core ROI after automatic HEIC stream selection: 57 images.
- Training device: CPU.
- Main GAN image size: 256.
- Small-data defaults used: batch size 16, no discriminator BatchNorm, real label smoothing at 0.9.

## Why the GAN Does Not Look Like the Target

### 1. The effective training set is too small

After filtering to images that actually contain a usable target, the GAN is learning from only about 50 images. That is too little for an unconditional image GAN to learn:

- target paper shape,
- black/white/blue/red/yellow ring ordering,
- perspective variation,
- arrows,
- lighting,
- background,
- stand/wheel/railing context.

The model can reduce loss by learning average color blobs and paper boundaries instead of the exact ring geometry.

### 2. The real data distribution is multi-modal

The cleaned ROI set still mixes several visual modes:

- frontal target,
- side-angle target,
- target plus stand,
- target plus floor/background,
- arrows in different positions,
- different crops and target scale.

An unconditional GAN has no label telling it which mode to generate. With few examples, it tends to average modes together. That is why the outputs become abstract blocks instead of one clean, stable target.

### 3. Some HEIC usable target streams are low-detail

The initial fixed `0:45` stream was wrong as a general rule. I changed the converter to select a HEIC stream by target-anchor score. That recovers some images, but some target-containing streams are still effectively small/low-detail thumbnails. Training at 256 does not create missing high-frequency ring detail if the decoded source is already limited.

### 4. The discriminator objective did not encode target geometry

The global discriminator accepted rough paper-like images too easily. Patch+LSGAN improved this: it pushed the generator toward black/white/blue/red/yellow regions. But a patch discriminator only checks local realism. It does not enforce:

- concentric circles,
- correct ring order,
- common center,
- roundness,
- target symmetry.

So Patch+LSGAN can produce locally plausible color regions while still failing the global target structure.

### 5. The generator architecture smooths structure

The upsample generator avoids checkerboard artifacts, but it is biased toward smooth blobs. That is visible in the best GAN result: large soft regions instead of crisp rings.

The transposed-convolution generator was also tested at 128. It produced checkerboard/mode-collapse artifacts and was worse.

### 6. Training longer did not fix the core issue

The 60-epoch Patch+LSGAN run improved over ordinary GAN, but did not converge to proper rings. The tighter target-core run made the data more focused, but it lost red/yellow center structure and generated mostly black/white/blue boundary forms.

That indicates the issue is not just "not enough epochs"; the task/data/model setup is underconstrained.

### 7. The product requirement is geometric

A usable archery target image has strict geometry: ordered concentric rings. GAN loss alone is a weak way to enforce that when data is tiny. A procedural generator can enforce the geometry directly; real-image augmentation preserves geometry from real samples. That is why the deliverable outputs are currently from:

- procedural target generation,
- real ROI augmentation.

## What Would Be Required for a GAN-Only Product

To make a GAN-only generator deliverable, the project would need one of these routes:

1. Collect a larger aligned dataset:
   - at least several hundred target-only crops,
   - ideally 1,000+ images,
   - consistent crop, target center, scale, and orientation.

2. Use a small-data GAN designed for this:
   - StyleGAN2-ADA or equivalent,
   - adaptive discriminator augmentation,
   - GPU training.

3. Add structure supervision:
   - train on target masks/ring layouts,
   - condition the generator on ring parameters,
   - add circle/edge/segmentation losses,
   - or generate geometric target first, then learn photo-style rendering.

4. Separate background from target:
   - first generate a clean target face,
   - then composite/augment into photo scenes.

## Recommended Product Path

For a product that must work now:

1. Use `outputs/effective_procedural_targets_256/` when clean target images are needed.
2. Use `outputs/effective_augmented_256/` when real-photo-like samples are needed.
3. Keep the GAN code and YAML matrix as an experiment harness, not as the product generator.

The GAN is useful as research evidence, but not as the current deliverable.
