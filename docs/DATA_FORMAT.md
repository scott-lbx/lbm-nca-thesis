# Data Format

How simulation output is stored, converted, and read by the training pipeline.
Three stages sit between the solver and the DataLoader:

```
HDF5 (one file per case)
  -> intermediate Zarr, flat per-case layout ("cases/{case_key}/...")
    -> unified Zarr, node/edge layout (topology in .zattrs or a sidecar graph.json)
```

The first two stages are produced by code in this repository
(`lbmnca.lbm.lbm_mrt_les.io.tes_writer`, `lbmnca.lbm.post_process.tes_h5_to_zarr`).
The third stage is read by `lbmnca.nca.tes.data.graph_index` and
`lbmnca.nca.tes.data.graph_dataset`, but the scripts that build it from the
intermediate Zarr (`make_zarr_cases_sidecar.py`, `build_mean_unified_zarr.py`,
referenced by `scripts/_src_build_zarr.py`) are not included in this
repository. Everything said about the unified format below is verified
against the reader code, not against a build script this repo contains.

## 1. Paths

Governed by `lbmnca.paths`, controlled by the `LBMNCA_DATA_ROOT` environment
variable (default `<repo>/data`):

| Root | Contents |
| --- | --- |
| `{LBMNCA_DATA_ROOT}/datasets/{name}/raw/{case_id}.h5` | Per-case solver output |
| `{LBMNCA_DATA_ROOT}/datasets/{name}/masks/{geometry_id}.png` | Grayscale geometry masks, ROI resolution |
| `{LBMNCA_DATA_ROOT}/zarr/{name}.zarr` | Converted Zarr (intermediate or unified, see below) |
| `{LBMNCA_DATA_ROOT}/zarr/{name}_stats.json` | Normalisation constants and dataset-scan statistics |

The internal project workspace this codebase was extracted from used
different root names (`dataset/`, `dataset_zarr/`) for the same roles; this
document describes the layout as this repository's own code produces and
reads it.

## 2. HDF5 case file

One file per simulation case, written by `tes_writer.TESCaseWriter`.

```
{case_id}.h5
├── (root attrs)              case metadata, see §2.2
├── /geometry/mask             bool   [H, W]           H=360, W=640
├── /geometry (attrs)          shape_h5 / shape_roi / shape_solver / downsample_factor
├── /g_vec                     fp32   [14]              present only if the caller supplied one
├── /convergence_info/
│     ├── kinetic_energy        fp32   [N_snap]
│     ├── mass_residual         fp32   [N_snap]
│     └── mass_residual_rel     fp32   [N_snap]
├── /trajectory/
│     ├── moments                fp16   [N_snap, 9, H, W]   lzf + shuffle
│     ├── lbm_steps              int32  [N_snap]
│     └── ctu                    fp32   [N_snap]
└── /mean_accum/                 present only if the solver run enabled the in-loop accumulator
      ├── segment_mean            fp32   [S, 9, H, W]
      ├── segment_n               int64  [S]
      ├── segment_step0/step1     int64  [S]
      └── segment_ctu0/ctu1       float64 [S]
```

### 2.1 Resolution and cropping

The writer crops the solver domain by the configured sponge and buffer
widths (`domain_zones` config: `sponge_{in,out,top,bot}`,
`buffer_{in,out,top,bot}`) to obtain the region of interest (ROI), which
equals the mask size (1920x1080 under the standard configuration), then
downsamples 3x with `cv2.INTER_AREA` for the moment fields and
`cv2.INTER_NEAREST` for the mask, producing the stored 640x360 (`W_ds x
H_ds`) grid. `H_ds = crop_h // 3`, `W_ds = crop_w // 3`
(`TESCaseWriter.DOWNSAMPLE_FACTOR = 3`).

Because the moment channels use area-averaging downsampling and the mask
uses nearest-neighbour, a cell that is partially solid at full resolution
can carry a fractional density in the stored moments while the nearest-
neighbour mask still labels it fluid. The intermediate-Zarr converter
(§4) corrects this by re-deriving solid fraction from the source mask PNG
and folding any partially-solid cell into the solid set before it computes
normalisation statistics or the static feature stack; the raw H5 mask does
not carry this correction.

### 2.2 Root attrs

Written by the case-generation caller (`case_meta` passed into
`TESCaseWriter.__init__`) plus a fixed set appended by
`TESCaseWriter.finalize()`. Fields consumed downstream (confirmed by
`tes_h5_to_zarr.py`, which reads them directly out of `f.attrs`):

| Field | Type | Notes |
| --- | --- | --- |
| `case_id`, `geometry_id` | str | Case and geometry identity |
| `re` | float | Reynolds number |
| `u_inlet_lu` | float | Inlet velocity, lattice units |
| `l_char_px` | int | Characteristic length, full-resolution pixels; read directly into `P_vec` (§below) and used to scale the SDF static channel |
| `ic_type` | str | `"cold"` / `"warm"` |
| `parent_case_id` | str | Present for warm cases |
| `window_type` | str | `"short"` marks a C2-2-style relation edge, excluded from global normalisation statistics and from its own `mean_flow_post_conv` (see §4) |
| `n_snapshots` | int | Set by `finalize()` |
| `converged` | bool | KE two-window convergence latch result (see `ARCHITECTURE.md` §7 for the criterion) |
| `convergence_ctu` | float | CTU at first convergence; `-1.0` if the latch never fired |
| `stop_reason` | str | One of `converged`, `converged_oscillatory`, `hard_cutoff`, `stability_failure`, `aborted`, `runtime_error` |
| `burnin_ctus`, `warmup_ctus` | float | Optional; explicit time-mean window start, overrides the fixed-fraction fallback (§4.2) when present |

Every field the H5-to-Zarr converter reads from these attrs is read with an
explicit optional/fallback marker in the source (`attrs.get(...)`), because
the attrs are a system boundary written by a solver that has evolved across
dataset generations; older files can be missing newer fields.

### 2.3 Mask convention

**`True` = solid.** Verified end to end, not taken from any prose
description:

- `lbmnca.lbm.lbm_mrt_les.public.mask.png_to_binary_mask` docstring and
  return value: `True = 固體` (solid).
- `lbmnca.lbm.lbm_mrt_les.utils.mask_utils.create_mask` docstring:
  "回傳布林陣列 ... True = 固體."
- The solver reads the mask with `if self.mask[i, j] == 1.0:` to skip
  collision and streaming at that cell (`LBM2D_MRT_LES.py::collide_stream`,
  region marked "固體節點不參與碰撞與串流").
- `tes_h5_to_zarr.py` reads `/geometry/mask` and documents it the same way.

`TESCaseWriter.__init__`'s own docstring is stale on this point: it
describes the `mask_data` argument as "free fluid = 1, solid = 0," which is
the inverse of what the code above actually implements. Do not use that
docstring as a source; use the mask utility functions and the solver
collision guard instead.

### 2.4 MRT 9-channel order

Fixed, solver-native, never permuted anywhere downstream:

| Channel | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Symbol | rho | e | eps | jx | qx | jy | qy | pxx | pxy |
| Meaning | density | energy moment | energy-squared moment | x-momentum | x-energy flux | y-momentum | y-energy flux | normal stress | shear stress |

Momentum is at channels 3 and 5 (not 1 and 2). `moments` in the trajectory
group and `mean_flow` / `mean_flow_post_conv` fields in the derived Zarr
formats (§4, §5) all use this same order.

**Velocity, vorticity, and pressure are derived quantities and must be
computed after denormalisation**, not on the normalised (tanh) values:
`ux = jx/rho`, `uy = jy/rho`, `p = rho * cs^2` with `cs^2 = 1/3`. This is
enforced structurally in `norm_frozen.macroscopic()`, which only accepts
raw moments, and is stated explicitly in `ARCHITECTURE.md` §3.

### 2.5 Snapshot schedule

Snapshots are recorded on a "staircase" CTU schedule
(`tes_writer.STAIRCASE_PHASES`): dense near t=0, coarsening with time.

| Phase | Delta (CTU) | Range (CTU) |
| --- | --- | --- |
| 0 | 0.2 | (0, 1] |
| 1 | 1.0 | (0, 5] |
| 2 | 1.0 | (5, 20] |
| 3 | 2.0 | (20, total_ctus] |

The schedule is shared between cold and warm cases and truncates
automatically at `total_ctus`; a warm case with a short `total_ctus`
therefore only exercises phases 0-1. A separate dense, fixed-spacing
schedule generator (`build_snapshot_schedule_dense`) also exists in the
writer module for warm-start datasets that need alias-free sampling of the
transient, but the staircase schedule is the one written into `case_meta`
as `schedule_type` for the datasets described in this document.

## 3. Normalisation (physics channels)

Two normalisation schemes exist in the codebase; only one is active in the
current write path.

**Active (`lbmnca.lbm.post_process.norm_frozen`, schema
`norm_frozen_v3`).** A fixed forward/inverse transform applied identically
at every stage, not recomputed per dataset:

```
out = tanh((raw - M_REF) / S)          in (-1, 1)
raw = atanh(clip(out, +/-CLIP)) * S + M_REF
```

`M_REF = [rho0, -2*rho0, rho0, 0,0,0,0,0,0]` with `rho0 = 1`, the D2Q9 MRT
rest-equilibrium state, derived analytically rather than measured. `S` is a
fixed per-channel scale (shared within the momentum pair, the energy-flux
pair, and the stress pair, to preserve rotational equivariance) chosen once
and frozen; it is not recalculated when a new dataset is added.
`ARCHITECTURE.md` §4 gives the channel groupings; `norm_frozen.py`'s module
docstring gives the full derivation and the two admissibility constraints
the constants were chosen to satisfy (density stays positive, velocity
stays subsonic). `tes_h5_to_zarr.py` imports `M_REF` and `S` directly from
this module (`nf.M_REF`, `nf.S`) for every write; the same constants are
copied into each dataset's `_stats.json` under `frozen_norm` for
provenance, and the module's own header states execution-time code is
authoritative over that copy.

**Superseded (`lbmnca.lbm.post_process.norm_utils`).** Describes a
three-stage scheme (baseline-deviation, then a per-dataset group-shared
rescale measured from data, then tanh). Several of its constants
(`CH_JX`, `CH_JY`, `NUM_CHANNELS`, `RESCALE_PERCENTILE`, `RHO_0`) are still
imported by `tes_h5_to_zarr.py`, but its data-driven rescale functions
(`compute_rescale_abs_max`, `raw_to_scaled`, `apply_group_max_rescale`) are
not called anywhere in the current write path (§4 below uses
`norm_frozen`'s fixed `S` directly). Treat this module as a historical
scheme still present for its shared constants, not as the active
normalisation.

Static channels (§4.2) and the conditioning vectors (`GVEC_SPEC.md`, and
`P_vec = [Re, U_inlet_lu, L_char_px]`) are normalised separately and are
**not** frozen: their z-score constants are measured from the training
population and must be reused (not recomputed) for validation and test
sets, or the resulting numbers are not comparable across splits.

## 4. Intermediate Zarr (`{name}.zarr/cases/{case_key}/...`)

Produced by `tes_h5_to_zarr.py`. One group per case, keyed by `case_id`
(fallback to a legacy key built from `geometry_id` when a file predates
that attribute). The module's own docstring calls this format "v2" /
mentions it as a step toward "v4"; the `version` string actually written
into the Zarr root attrs and into `_stats.json` is `"tes_v4"`. Treat the
string as a version tag rather than as evidence this is the node/edge
format of §5 -- it is not; the group layout is flat.

```
{name}.zarr
├── .zattrs   {version: "tes_v4", H: 360, W: 640, n_cases, snapshot_mode, scope: "mean_only", ...}
└── cases/{case_key}/
      ├── snapshots                    fp16   [T, 9, H, W]     T depends on --snapshot-mode (see below)
      ├── static                       fp32   [4, H, W]        SDF_tanh / upstream_z / X01 / Y01
      ├── g_vec                        fp32   [14]             raw, see GVEC_SPEC.md
      ├── p_vec                        fp32   [3]               raw [Re, U_inlet_lu, L_char_px]
      ├── delta_ctus                   fp32   [T-1]             np.diff(ctu_values)
      ├── mean_flow_post_conv          fp32   [9, H, W]        target field for the mean module (absent on "short"-window cases)
      ├── mean_flow_half1/half2        fp32   [9, H, W]        two independent half-window means, present only when the H5 has /mean_accum; the sole basis for a GT self-noise estimate
      └── mean_flow_post_conv_running  fp32   [9, H, W]        present only when the H5 has /mean_accum and the running accumulator is not already the selected source
```

`--snapshot-mode` (default `ic_only`) controls `T`: `"none"` stores no
`snapshots` array at all, `"ic_only"` stores exactly the first frame
(the cold-start initial condition, a state of zero deviation from
equilibrium), `"full"` stores the whole trajectory. `"full"` is explicitly
flagged in the source as producing snapshots the frozen normalisation
constants were not calibrated for (transient peaks run 4-5x the amplitude
of the time-mean field) -- values can saturate the tanh range and the
inverse transform is then lossy. The datasets this document otherwise
describes were converted with `ic_only`.

**Solid cells in every stored field are overwritten to `M_REF`** (not left
at their raw solver value) before normalisation, using the corrected solid
mask from §2.1. This removes the fluid/solid interface discontinuity that
raw moments would otherwise carry, at cells whose geometry is already
represented by the static SDF channel.

### 4.1 Static channels

`static[c]` for c in 0..3:

| c | Field | Method |
| --- | --- | --- |
| 0 | SDF | `tanh(SDF_px / (L_char_px / 3))`, per-case, computed from the corrected mask; zero at the fluid/solid boundary |
| 1 | upstream | z-score against a dataset-measured mean/std (`upstream_mean`, `upstream_std` in `_stats.json`) |
| 2 | X | `(x + 0.5) / W`, 0 = inlet, 1 = outlet |
| 3 | Y | `(y + 0.5) / H`, 0 = bottom, 1 = top |

### 4.2 Time-mean window

`mean_flow_post_conv` is a time-weighted mean of the raw moments over a
window, then normalised. The window's **start** is resolved by
`_resolve_mean_start_ctu` with this priority: an explicit `burnin_ctus`
config value (if the H5 has `/mean_accum` and the attr is present and
non-negative) takes precedence over a fixed fraction
(`MEAN_ACCUM_START_FRAC = 0.2` of the trajectory's total CTU, only when the
H5 has `/mean_accum`), which in turn takes precedence over the
convergence-latch CTU (`convergence_ctu`, the only option for H5 files
without an in-loop accumulator). The window's **weighting** is controlled
by `TES_MEAN_WINDOW` (`square`/`hann`/`hann_sq`, default `square`) and its
**source** by `TES_MEAN_SOURCE` (`snapshot`/`running`, default `running` --
i.e. the in-loop accumulator is preferred over averaging the sparse
recorded snapshots when both are available). Whichever basis and source
were actually used are recorded per case as `mean_window_basis`,
`mean_window_start_ctu`, `mean_flow_source`.

### 4.3 `{name}_stats.json`

Written once per dataset conversion. Sections, verified against
`_build_stats_json`:

| Key | Content |
| --- | --- |
| `frozen_norm` | Copy of the `norm_frozen` constants (`M_REF`, `S`, `ATANH_A`, `CLIP`, `R`, the admissible box) for provenance |
| `physics_norm` | Back-compat alias: `M_REF`, `S_combined` (identical to `frozen_norm`'s values) |
| `compliance` | Per-channel report of whether every fluid-masked mean field in the dataset fits inside the frozen scale without clipping; conversion raises if any channel clips |
| `static_norm` | Method + measured constants for each of the four static channels |
| `cond_stats` | Mean/std used to z-score `g_vec` (14D) and `p_vec` (3D) |
| `vis_range` | Fixed display-range constants for colour-mapping (velocity/vorticity/pressure), orthogonal to the tanh normalisation above |
| `stats_scope` | Provenance: which dataset role produced these statistics, how many cases, whether they were borrowed via `--stats-from` |

Only a dataset declared `--dataset-role train` may compute its own
`static_norm` / `cond_stats`; `val` and `test` datasets must pass
`--stats-from` pointing at a train dataset's `_stats.json`, or the
converter refuses to run. Physics normalisation is frozen and is
unaffected by dataset role.

## 5. Unified Zarr (node/edge layout)

This is the format `GraphIndex` and the training-side `Dataset` classes
(§6) actually read: `graph.GraphIndex.for_zarr` opens
`{name}_unified.zarr` and reads its topology from `.zattrs["topology"]`
(the single-file / "unified" layout), or, for an older on-disk layout not
produced anywhere in this repository, from a sidecar
`{name}_v4_graph.json` next to `{name}_v4.zarr`. Both layouts carry the
same topology schema; only the container differs.

```
{name}_unified.zarr
├── nodes/{node_id}/
│     ├── static       fp32  [4, H, W]
│     └── mean_flow    fp32  [9, H, W]     the node's converged, time-averaged state
├── edges/{edge_id}/
│     ├── snapshots    fp16  [K, 9, H, W]  K = 1 under ic_only (see §4)
│     └── ctu          fp32  [K]
└── .zattrs["topology"]  {version, format, dataset_id, stats_space, grid, nodes: [...], edges: [...]}
```

`GraphIndex` recognises two `topology.format` values, and treats them as
carrying **different normalisation semantics, not interchangeable**:

| `format` | Contains | Normalisation |
| --- | --- | --- |
| `tes_unified_v1` | Adds a `postconv` window array and `mean_flow_running` per node (older generation) | `S_combined` measured from data |
| `tes_unified_mean_v1` | Mean-only: no `postconv`, no transient snapshots beyond the IC frame | Frozen constants (§3) |

`GraphIndex.is_mean_only` is `True` only for the second. The datasets this
codebase's frozen-normalisation and IC-only-snapshot conventions (§3, §4)
describe are `tes_unified_mean_v1`.

### 5.1 Node / edge semantics

**Node = an attractor.** A converged, time-averaged flow state for one
geometry + Reynolds number, independent of how the solver arrived at it.
**Edge = a transit.** The closed interval from a source node's converged
state (used as the initial condition) through a target geometry change, up
to the target's own convergence. This is stated as the governing
distinction in `graph_index.py`'s module docstring and is the reason node
and edge data do not duplicate each other: nodes carry no trajectory,
edges carry no static geometry (`static` and `g_vec`/`p_vec` belong to the
node; edges reference their endpoint nodes for that information).

### 5.2 Topology fields

`.zattrs["topology"]` (or the sidecar `graph.json`) is a plain dict with
top-level keys `version` (must be `"tes_v4"`), `dataset_id`, `stats_space`,
`grid`, `format`, `role_source`, `nodes` (list), `edges` (list).
`GraphIndex` stores `grid` as-is (`self.grid = graph["grid"]`) without
destructuring it further, so its internal sub-keys are not independently
verified by any code path in this repository; treat it as an opaque
dict describing the field resolution and channel count. Per-record fields
below are those `graph_index.py` and `graph_dataset.py` actually key into:

**Node**

| Field | Read by |
| --- | --- |
| `node_id` | primary key |
| `mask_id` | `EdgeMeanDataset` (case metadata passthrough) |
| `ic_type` | `EdgeMeanDataset` (case metadata passthrough) |
| `tree_depth` | `GraphIndex.nodes_by(depth=...)` |
| `external` | `GraphIndex.nodes_by(external=...)`; marks a node whose source case lives in a different (cold-root) dataset |
| `canonical_edge_id` | `GraphIndex.canonical_edge_of`: the edge that defines this node's post-conv state |
| `g_vec` | `GraphIndex.g_vec` -- raw 14D, see `GVEC_SPEC.md` |
| `p_vec` | `GraphIndex.p_vec` -- raw `[Re, U_inlet_lu, L_char_px]` |
| `cube_geom` | optional; frozen param-PCA cube coordinate, read via `.get()` by `graph_split.build_far_src_map`, falls back to `g_vec`-space distance when absent |
| `split` | present only in the older, non-unified layout (`tes_unified_v1` with an independent `graph.json`); the `tes_unified_mean_v1` unified format has **no `split` key at all** -- role (train/val/test) is assigned by which `zarr_paths` bucket a config puts the whole file in, not stored per node |

**Edge**

| Field | Read by |
| --- | --- |
| `edge_id` | primary key |
| `src_node` | `GraphIndex.src_of`; `None` marks a cold-start edge |
| `dst_node` | `GraphIndex.dst_of` |
| `ic_source` | `"parent_mean"` or `"cold_start"` |
| `edge_type` | observed values in this codebase: `chain`, `re`, `knn`, `clique`, and the reversed variants `geom_backward` / `re_backward` |
| `window_type` | `"full"` or `"short"`; short windows are excluded from global normalisation statistics and get their `mean_flow` quality-checked against their canonical full-window counterpart (§4's `attractor_qa_ratio`) rather than trusted directly |
| `traj_ref` | resolves a case-id collision alias to the edge that actually owns the stored trajectory |
| `attractor` | optional dict with a `ratio` key -- the short-edge self-calibrated consistency check (`max_attractor_ratio` filtering in `GraphIndex.edges_by`) |
| `split` | same caveat as the node field above -- absent in the unified layout |

Not verified in this repository: `n_postconv_frames`, which appears in an
external specification document for an older node schema but is not read
by any code path in this repo (the current `tes_unified_mean_v1` format
has no `postconv` array for it to describe).

### 5.3 Dataset scale

The internal project's specification document
(`docs/SSOT/DATA_ZARR_D系列規格.md`, dated 2026-08-05) reports node/edge
counts for the six datasets this repository's own conversion driver
(`scripts/_src_build_zarr.py`) names identically as
`TRAIN_HOST`/`TRAIN_OTHERS`/`EVAL_SETS`/`TEST_SETS`:

| Role | Datasets | Nodes | Canonical edges | `parent_mean` (warm continuation) edges | `cold_start` edges |
| --- | --- | ---: | ---: | ---: | ---: |
| train | `D0-1_Cold_M40C40` + `D1-1_Cold_M400C400` + `D2-1_Warm_M200C200R24S60` | 571 | 571 | 173 | 398 |
| val | `DE0-1_Cold_M12` + `DE1-1_Warm_M48R12` | 58 | 58 | 25 | 33 |
| test | `DT_Stress` | 59 | 59 | 9 | 50 |

The 571-node train figure is independently corroborated inside this
repository: `norm_frozen.py`'s calibration note states the frozen scale
constants were calibrated by scanning "the full population of 571 nodes"
across exactly those three train datasets. The val and test figures are
not independently corroborated by any code in this repository (the scripts
that would regenerate them are not included, per the note at the top of
this document) and should be attributed to that specification document
rather than to this repository's own code.

Node and edge counts are always equal (one canonical edge per node) in
this corpus; citing "edge count" as a distinct figure from node count is
only informative when it is broken down into `parent_mean` vs `cold_start`
as above.

On-disk size: converting to the mean-only format described in §4-§5
(dropping `postconv`, transient snapshots beyond the IC frame, and the
running-mean duplicate array) reduces total size for these six datasets
from about 164.5 GB to about 14 GB, per `scripts/_src_build_zarr.py`'s own
module docstring.

## 6. Training-side `Dataset` classes

`lbmnca.nca.tes.data.graph_dataset` defines four `torch.utils.data.Dataset`
subclasses, all reading from a single unified Zarr plus its `GraphIndex`:

**`_V4Base`** -- shared base, not used directly. Provides lazy zarr-handle
opening (deferred past DataLoader worker fork), the vertical-flip
augmentation (`_maybe_vflip`, train-only; valid because the solver's
left-to-right inlet/outlet geometry makes only a y-mirror physically
admissible, not a full rotation), and two read helpers:
`_node_cond(node_id)` (returns `static` + `g_vec` + `p_vec`) and
`_mean_flow(node_id)` (returns the node's `mean_flow` tensor).

**`EdgeMeanDataset`** -- one edge per item. Reads `ic_source="parent_mean"`
edges by default (cold-start edges must be requested as a separate,
homogeneous dataset instance -- mixing the two inside one batch would push
the whole batch onto the trainer's cold-start code path). For a
`parent_mean` edge, `IC = mean_flow(src node)`, `target =
mean_flow(dst node)`, condition = `dst node`'s `static`/`g_vec`/`p_vec`.
For a `cold_start` edge, `IC` is instead the edge's own stored
first-frame `snapshots[0:1]`. Supports an optional `src_override` map used
by the three-arm ablation (`far_edge`/`rand_edge`) to substitute a
different node's mean field as the IC while leaving the target and
condition untouched.

**`NodePairDataset`** -- an arbitrary `(IC node, target node)` pair, not
constrained to an existing graph edge. Exists because `EdgeMeanDataset`
only exposes the graph's canonical edges (9 usable pairs for `DT_Stress`),
while evaluation wants every node in a test set usable as either an IC or
a target regardless of whether that specific transition was ever
simulated (59 pairs for `DT_Stress`). Condition is always taken from the
target node.

**`EdgeChainMeanDataset`** -- multi-stage pushforward along a lineage
chain of `chain_len` consecutive edges. Stage 1's IC is the chain's first
node's `mean_flow`; each subsequent stage's IC is the model's own output
from the previous stage (a hand-off, mirroring deployment), and every
stage is supervised against its own node's `mean_flow`. Built from chains
enumerated by the module-level `enumerate_chains` helper, which only walks
edges already restricted to one split.

Two matching collate functions exist:
`collate_graph_batch` (for `EdgeMeanDataset` / `NodePairDataset`, which
stacks whichever of a fixed key set is present in a sample rather than
assuming all keys exist) and `collate_chain_batch` (for
`EdgeChainMeanDataset`, which stacks the fixed-length per-stage tensors
directly).

`lbmnca.nca.tes.data.graph_split.split_graph_datasets` is the entry point
that turns a list of unified Zarr paths plus a declared `role` into
`EdgeMeanDataset` instances bucketed into train/val/test; it is the layer
that enforces "one unified Zarr file is entirely one role" for the
`tes_unified_mean_v1` format (§5.2).
