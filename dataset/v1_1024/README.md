# v1_1024

Immutable real-image dataset for GAN training. Each accepted source contributes exactly one deterministically decoded, target-cropped, Lanczos-resized 1024 x 1024 PNG.

- train: 80
- val: 0 (intentionally unused when zero; ADA does not require validation data)
- test: 20
- split unit: short contiguous filename capture groups; a group never crosses splits
- random offline augmentation: disabled
- test images: never used by the GAN training loader or ADA

Evidence boundary: the current sources are continuous photographs of the same physical target and background. The test directory is an internal same-scene holdout, not an independent real-world test set. A future external test must use a new session/background.

Training-time augmentation belongs in the discriminator path and is generated in memory. Do not add flipped, recolored, cutout, or Copy-Paste variants to this directory.
