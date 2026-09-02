# Reproducibility

What can be reproduced exactly, what can be reproduced approximately, and what
cannot. Read this before comparing your numbers against the thesis.

---

## Three tiers

**Exact.** Running the released checkpoint on a given input. The weights in
`checkpoints/` are the ones the reported numbers came from, the normalisation
constants are frozen, and the loader refuses a checkpoint whose recorded
architecture disagrees with the configuration. Given the same input field you
should get the same output, up to floating-point differences between GPU
architectures.

**Approximate.** Retraining from a regenerated dataset. The configuration fixes
`seed: 42`, but `determinism_mode: warn_only` — cuDNN autotuning, mixed
precision, and multi-worker data loading are all left enabled because turning
them off would have made training impractically slow. Two runs of the same
configuration on the same machine will not be bitwise identical, and a run on
different hardware will differ more. Expect the same qualitative conclusions and
metric values within run-to-run noise, not identical numbers.

**Not reproducible bit-for-bit.** The dataset itself. Regeneration is seeded, so
the same seed and the same population size give the same layouts; but the
simulation runs on your GPU, and lattice-Boltzmann accumulation over hundreds of
thousands of steps amplifies floating-point ordering differences. The statistical
properties will match; individual case fields will not be identical.

**Measure the noise floor before attributing anything to an effect.** The
canonical arm `main_v4` and the ablation arm `abl_use_v4` have identical
configurations. The difference between them is training noise. Any ablation
effect smaller than that gap is not an effect.

---

## Reproducing an inference result

```bash
python scripts/run_inference.py \
    --checkpoint checkpoints/main_v4.pth \
    --config configs/nca/main.yaml
```

This needs no dataset. If it prints the parameter count and the training step,
your installation is correct.

Two settings must match training or the reading is invalid:

- `psi_lowpass_sigma` must be `0.8`. The low-pass filter applies identically at
  training and at deployment, so evaluating with another value is a
  configuration mismatch, not a variant.
- The normalisation statistics must be the ones in `checkpoints/norm/`. These
  are frozen constants, not recomputed from whatever data is loaded.

## Reproducing a training run

1. Regenerate the datasets named in `configs/nca/main.yaml` at the sizes given in
   [DATA_GENERATION.md](DATA_GENERATION.md). Smaller datasets will train but will
   not reproduce the numbers.
2. Point `LBMNCA_DATA_ROOT` at them.
3. Run:

```bash
python scripts/train.py --config configs/nca/main.yaml
```

The wrapper checks that every configured Zarr path exists before starting, so a
missing dataset fails immediately rather than partway through the first epoch.

## Reproducing the ablation study

`configs/nca/ablations/` holds the arms of the data-condition factorial. Each
varies one thing about how the training pair's starting field is chosen:

| Arm | Starting field |
| --- | --- |
| `abl_use_v4` | The true nearest neighbour's converged field (A✓B✓) |
| `abl_far_v4` | The furthest available pairing instead (A✓B✗) |
| `abl_noise_v4` | The true neighbour, with noise added (A✗B✓) |
| `abl_farnoise_v4` | Furthest pairing and noise (A✗B✗) |
| `abl_rand_v4` | Random pairing, a robustness check on the pairing factor |
| `abl_no_v4` | A zero field. This is the origin, not a factorial cell, and does not enter the effect calculation |

`main_v4` is a technical replicate of `abl_use_v4`; their difference is the noise
floor.

Two paradigm baselines are also included: `base_modern` (sequential iteration)
and `base_steady` (direct mapping). Their parameter counts and per-step costs are
of a different order from the surrogate, so any comparison must state both.

Only `main_v4` and `base_modern` ship as weights. The rest are configurations you
can train yourself; `base_modern` is included because the live demo needs it to
produce a cold-start initial condition.

## Things that will make your numbers wrong

- **Recomputing normalisation statistics on a validation set.** Pass
  `--stats-from` the training set's statistics when converting validation and
  test data. Otherwise the validation distribution leaks into the normalisation
  and every validation number comes out optimistic.
- **Refitting the PCA design space after cases exist.** The transform is frozen
  for a reason: every case's coordinates are expressed in it.
- **Comparing arms across generations.** Only the `_v4` arms in
  `configs/nca/ablations/` are mutually comparable. Earlier generations of the
  same names optimised different objectives and are not included here.
- **Evaluating a free-running operator.** With automatic stopping disabled the
  field keeps evolving and visibly degrades, while the kinetic-energy criterion
  stays below its threshold and never signals it. See `ARCHITECTURE.md` §7.
- **Reading the comfort overlay as physics.** The lattice-to-physical velocity
  scale is a placeholder. The renderer marks the scale as uncalibrated on screen.

## Provenance recorded in a run

Each training run records the git commit, the seed, the completed step count and
the PyTorch version alongside its checkpoints, and each evaluation reading
records the checkpoint path, the configuration path, the commit, the timestamp
and the hardware. When reporting a number, report which run produced it.
