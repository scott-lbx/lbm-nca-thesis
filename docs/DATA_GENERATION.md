# Regenerating the dataset

The dataset is not distributed with this repository. It is large, and every part
of it is reproducible from the code here. This document is the procedure.

Read [DATA_FORMAT.md](DATA_FORMAT.md) first if you want to know what the
generated files contain; this document is about how to produce them.

---

## Before you start

**This needs a GPU for a long time.** The solver runs on a 2560 × 1440 lattice
with a multiple-relaxation-time collision and a Smagorinsky closure, and each
case is simulated until a two-window kinetic-energy criterion declares
convergence. The full dataset used in the thesis is 670 nodes and 748 edges
across four datasets; the converted Zarr occupies roughly 10 GB, and the
intermediate HDF5 trajectories considerably more.

Wall-clock cost per case depends on the Reynolds number and on how quickly the
case converges, so a single number would be misleading. Measure it on your own
hardware with the smoke configuration in the next section before planning a full
run.

Everything is written under `LBMNCA_DATA_ROOT` (default `<repo>/data`):

```
$LBMNCA_DATA_ROOT/
├── design_space/            population and the frozen PCA transform
├── datasets/<name>/         masks, per-case configs, raw HDF5
└── zarr/                    converted training format
```

Set it somewhere with room:

```bash
# Windows
set LBMNCA_DATA_ROOT=D:\lbmnca-data
# POSIX
export LBMNCA_DATA_ROOT=/scratch/lbmnca-data
```

## Verify your environment first

Before committing days of GPU time, generate a handful of cases end to end. If
this completes and the resulting Zarr loads, the pipeline is wired correctly on
your machine.

```bash
python -m lbmnca.morphospace.param_pca.generate --n-pop 20000 --seed 0
python -m lbmnca.morphospace.param_pca.cube
python -m lbmnca.morphospace.dataset.sample_cold --dataset-name SMOKE --n-cases 4 --seed 0
python -m lbmnca.lbm.pipeline.batch --dataset-name SMOKE
python -m lbmnca.lbm.post_process.tes_h5_to_zarr --dataset-dir $LBMNCA_DATA_ROOT/datasets/SMOKE --dataset-role train
```

A small population (20 000 rather than 1 000 000) is fine for a smoke test, but
**do not train on a dataset built from it**: the PCA design space is fitted to
the population, so a small one gives a different, poorly conditioned space.

---

## The six steps

### 1. Sample the layout population

```bash
python -m lbmnca.morphospace.param_pca.generate --n-pop 1000000 --seed <SEED>
```

Rejection-samples one million legal rectangular building layouts. Pure NumPy, no
GPU, a few minutes. Writes `design_space/population_big.npz`.

Legality is checked by `lbmnca.morphospace.param_pca.legality`: no overlaps, no
out-of-bounds buildings, minimum inter-building gap respected.

### 2. Fit and freeze the design space

```bash
python -m lbmnca.morphospace.param_pca.cube
```

Runs PCA on the population's geometric features and writes `cube4d.npz`, the
frozen transform defining the four-dimensional design space: three geometric
principal components plus a standardised Reynolds number.

**This transform must stay frozen for the rest of the pipeline.** Every case is
positioned in this space, so refitting it after cases exist silently invalidates
their coordinates. If you regenerate the population, regenerate everything
downstream too.

### 3. Sample cases

Cold cases are independent starting points; warm cases form chains from a cold
root, each a small edit of its parent. The chain structure is what the surrogate
is trained on, so both are needed.

```bash
python -m lbmnca.morphospace.dataset.sample_cold \
    --dataset-name D1-1_Cold_M400C400 --n-cases 400 --seed 0

python -m lbmnca.morphospace.dataset.sample_warm \
    --dataset-name D2-1_Warm_M200C200R24S60 \
    --cold-dataset D1-1_Cold_M400C400 \
    --n-new-masks 200 --n-roots 24 --n-shortcuts 60 --seed 0
```

The dataset names encode their own parameters: `M200C200R24S60` means 200 new
masks, 200 cases, 24 minimum-spanning-tree roots, 60 shortcut simulations. Keep
the convention if you want the configuration files to keep working unchanged.

Cold sampling covers the design space on a grid so the population is spread
rather than clustered. Warm sampling builds a minimum spanning tree over the
sampled points, so each warm case is genuinely close to its parent. Both write
`cases.json` and the mask PNGs.

For the stress-test holdout used to evaluate out-of-distribution behaviour:

```bash
python -m lbmnca.morphospace.dataset.gen_stress_cases --dataset-name DT_Stress
```

### 4. Simulate

```bash
python -m lbmnca.lbm.pipeline.batch --dataset-name D1-1_Cold
```

This is the expensive step. For each case it writes a per-case YAML (physical
constants, warm-up and averaging windows, sponge geometry), runs the solver to
convergence, and writes an HDF5 trajectory.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--dry-run` | Generate configs and report the plan without simulating |
| `--max-cases N` | Stop after N cases, for a partial run |
| `--no-replace` | Do not attempt automatic replacement of failed warm cases |

Warm cases are scheduled breadth-first from their cold roots, because a warm case
cannot start until its parent has converged. If a warm case fails, the scheduler
can substitute a nearby alternative so the chain is not truncated.

**Finishing is not the same as being accepted.** Every case passes a physics
validation gate (`lbmnca.lbm.validation.case_physics`) before it enters the
dataset. A case that ran to completion but violates the admissibility box is
rejected rather than silently kept.

### 5. Convert to Zarr

```bash
python -m lbmnca.lbm.post_process.tes_h5_to_zarr \
    --dataset-dir $LBMNCA_DATA_ROOT/datasets/D1-1_Cold \
    --dataset-role train \
    --snapshot-mode ic_only
```

Downsamples the fields by 3× to 640 × 360, applies the frozen normalisation,
computes the static channels and `G_vec`, and writes the per-case Zarr plus a
`_stats.json`.

`--dataset-role` is mandatory and is one of `train`, `val`, `test`. For
validation and test sets you must reuse the training set's normalisation
constants:

```bash
python -m lbmnca.lbm.post_process.tes_h5_to_zarr \
    --dataset-dir $LBMNCA_DATA_ROOT/datasets/DE0-1_Cold \
    --dataset-role val \
    --stats-from $LBMNCA_DATA_ROOT/zarr/D1-1_Cold_stats.json
```

Recomputing statistics on the validation set would leak its distribution into the
normalisation and make every reported validation number optimistic.

### 6. Build the unified node/edge dataset

```bash
python -m lbmnca.lbm.post_process.make_cases_sidecar --dataset D1-1_Cold
python -m lbmnca.lbm.post_process.build_unified_zarr --dataset D1-1_Cold --execute
```

Separates converged attractors (nodes) from transitions between them (edges) and
writes the `graph.json` that the training loader reads. Without `--execute` the
second command reports what it would build and stops.

The result is what `configs/nca/main.yaml` points at through `${ZARR_ROOT}`.

---

## Datasets used in the thesis

The configuration files reference these names. Reproduce the sizes to reproduce
the reported numbers; smaller runs will train but will not match.

| Name | Role | Content |
| --- | --- | --- |
| `D0-1_Cold_M40C40` | train | 40 layouts, cold starts |
| `D1-1_Cold_M400C400` | train | 400 layouts, cold starts |
| `D2-1_Warm_M200C200R24S60` | train | 200 layouts, warm chains, 24 roots |
| `DE0-1_Cold_M12` | val | 12 layouts, cold |
| `DE1-1_Warm_M48R12` | val | 48 layouts, warm, 12 roots |
| `DT_Stress` | holdout | Stress cases for out-of-distribution evaluation |

---

## If something goes wrong

**The solver diverges.** Check the Mach number and the relaxation time in the
generated per-case YAML. The feasibility check
(`lbmnca.lbm.config_gen.config_utils.feasibility`) should reject infeasible
combinations before simulation; a divergence that gets past it is worth
reporting as a bug.

**`G_vec` cannot be computed.** The converter raises rather than writing a zero
vector, deliberately: a zero `G_vec` is silently wrong conditioning, which is far
worse than a stopped run. Check that the mask PNG for that case exists.

**Statistics differ between runs.** They should, slightly, if you changed the
seed or the population. They must not differ within one dataset generation. If
they do, you have regenerated the population without regenerating the cube.

**Caches.** The feature caches under `design_space/` are keyed by dataset, so
pointing a script at a different dataset will not reuse the wrong cache. Pass
`--force` to recompute anyway.
