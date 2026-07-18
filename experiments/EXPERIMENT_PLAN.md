# Arrow-target GAN canary plan

All experiment metadata and generated artifacts live in the same version directory.
Every arm uses the same 1024 px G48/D16 architecture, seed 42, data preprocessing,
DiffAugment policy, batch size, and 25-epoch budget. Only the learning-rate setting
changes, so the comparison remains interpretable.

## Execution order

0. `EXP-GAN-1024-000-local-smoke`: one CPU step, pipeline evidence only.
   While GPU access is unavailable, run `010/011/012` as matched four-step
   local learning-rate probes; these only screen for immediate instability.
1. `EXP-GAN-1024-001-base`: G LR 2e-4, D LR 1e-4.
2. `EXP-GAN-1024-002-midlr`: G LR 5e-4, D LR 1e-4. Run only after arm 001 is reviewable.
3. `EXP-GAN-1024-003-highlr`: G LR 1e-3, D LR 5e-4. Treat as a high-risk canary.
4. `EXP-GAN-1024-004-base-long100`: resume the base arm from epoch 20 to total
   epoch 100 with early stopping disabled. This tests whether recognizable shapes
   emerge only after a substantially larger epoch budget.

Run command:

```bash
python -u -m model.train_gan --config experiments/<version>/config.json
```

After each arm, generate a fixed 64-image set and evaluate the paired discriminator.
Review target geometry, arrow presence, physical arrow shape, identifiable impact point,
diversity, and the four discriminator groups (`real_as_real`, `real_as_fake`,
`fake_as_fake`, `fake_as_real`) before starting the next arm.

Early stopping is conservative: evaluate the fixed seed-42 sample every five epochs,
keep the lowest target-structure-error checkpoint, and stop only after epoch 20 when
three evaluations fail to improve. Periodic states may be retained during an active run
for recovery, but after completion they are consolidated to final resumable states plus
selected visual milestones. Arrow quality still requires visual review and must not be
replaced by a ring-only metric.
