# EXP-GAN-1024-001-base

- Status: completed with early stop at epoch 20
- Seed: 42
- Architecture: 1024 px, G48, D16
- Learning rates: G 2e-4, D 1e-4
- Budget: 25 epochs
- Early stop: from epoch 20, patience 3 evaluations, min delta 0.005
- Hypothesis: conservative baseline should establish whether the compact model can learn target structure without immediate instability.

## Results

- Final loss D/G: 0.5216 / 1.4225
- Tail-five loss D/G: 0.5150 / 1.1453
- Final D(real)/D(fake): 0.6478 / 0.3041
- Tail-five D(real)/D(fake): 0.7243 / 0.3602
- Final discriminator: real accuracy 96.04%, fake accuracy 100%, balanced accuracy 98.02%
- Real images misclassified as fake: IMG_4014 (0.4492), IMG_4073 (0.4324),
  IMG_4090 (0.0618), IMG_4102 (0.3350), where values are P(real)
- Generated images misclassified as real: none
- Final generated set: 64 images from the epoch-20 EMA checkpoint
- Final shape review: no recognizable target or arrow shape

The run is numerically finite, but the discriminator leads the generator at the
endpoint. A separate long-run version resumes this state to total epoch 100 before
making the larger-epoch conclusion.
