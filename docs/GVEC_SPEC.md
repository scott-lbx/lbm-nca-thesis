# G_vec Specification

`G_vec` is a 14-dimensional geometric conditioning vector computed from a
binary occupancy mask alone -- no simulation, no Reynolds number, no
solver run. It is the geometry half of the model's conditioning vector
(`ARCHITECTURE.md` §2: `cond_dim = 17 = G_vec (14) + P_vec (3)`).

Source of everything in this document:
`lbmnca.morphospace.physics_proxy.gvec_api` (constants `GVEC_ACTIVE_INDICES`,
`GVEC_FEATURE_NAMES`, `GVEC_FEATURE_SPEC`, `GVEC_HONEST_NAMES`,
`GVEC_INERT_DIMS_C_SERIES`) and `lbmnca.morphospace.physics_proxy.features`
(the 22-feature superset `G_vec` is drawn from). `gvec_api.py`'s own
docstring is the origin of the per-dimension honesty table in §2; nothing
in that table has been softened here.

## 1. What it is and where it comes from

`features.py` computes 22 scalar features from a mask (`compute_all`),
grouped A through F: A (obstacle spatial distribution), B (flow-path
geometry), C (blockage and exposure), D (wall-distance statistics), E
(cross features that depend on the simulation setup, not geometry alone),
F (inlet/outlet boundary features). `G_vec` selects 14 of those 22
(`GVEC_ACTIVE_INDICES`), producing a purely geometric vector: `E1_re_eff`
and `E2_blockage_corrected_velocity` are excluded by name in the module
docstring specifically because they depend on the physical setup (a
Reynolds number, an inlet velocity), not on the mask alone -- keeping them
out means `G_vec` describes the layout and nothing else. The remaining six
excluded features (`B3_void_connectivity`, `C4_ud_bcr_ratio`,
`D2_std_wall_distance_px`, `E3_canyon_ar_mean`, `F1_inlet_fluid_fraction`,
`F2_inlet_entropy`) are not selected into the active subset; the source
does not give a further stated reason for excluding these six
specifically beyond E1/E2, and none is asserted here -- **not verified**.

The mask convention for every function in this module is the same as
`DATA_FORMAT.md` §2.3: 1 (`True`) = solid, 0 (`False`) = fluid.

## 2. Per-dimension table

`honest` is the name the source recommends using; `code` is the
identifier actually used in `G_vec`'s array position and in existing data
files. Every row has `claims_physics = False` in the source -- see §3.

| # | `code` | `honest` | scope | computed | verdict |
| - | --- | --- | --- | --- | --- |
| A1 | `A1_min_inter_obstacle_gap_px` | `min_obstacle_boundary_distance_px` | bbox | Global minimum of the pairwise nearest-neighbour Euclidean distance between the boundary points of each solid connected component (via KDTree). If there are 0-1 obstacles, returns `max(H_bbox, W_bbox)` as a sentinel, not a gap. | partially matches its name |
| A2 | `A2_mean_inter_obstacle_gap_px` | `mean_nearest_neighbour_obstacle_distance_px` | bbox | For each obstacle, its nearest-neighbour distance to any other obstacle, averaged over obstacles -- **not** the mean over all pairs. Same 0-1-obstacle sentinel as A1. | partially matches its name |
| A3 | `A3_obstacle_count` | `solid_connected_component_count` | bbox | Count of 4-connected solid components (touching bodies merge into one). | matches its name |
| A4 | `A4_mean_obstacle_area_ratio` | `mean_obstacle_area_over_bbox_area` | bbox | Mean, over obstacles, of (obstacle pixel count / bounding-box area). The denominator is the bounding-box crop area, **not** the full LBM domain and **not** a site/plot area -- this is not a building coverage ratio. | partially matches its name |
| A5 | `A5_obstacle_size_cv` | `obstacle_size_coefficient_of_variation` | bbox | std/mean of the per-obstacle area ratio (scale-invariant, so the choice of denominator above does not affect it). | matches its name |
| B1 | `B1_throat_width_px` | `min_free_cross_section_px` | bbox | Minimum, over x-columns, of the fluid pixel count in that column -- equivalently, `H_bbox` minus the widest solid cross-section. Multiple parallel gaps in the same column are summed together, so this is **not** the width of any single gap. | does not match its name |
| B2 | `B2_tortuosity` | `one_plus_geodesic_over_domain_width` | full | Geodesic-BFS (`MCP_Geometric`) shortest path from x=0 to every reachable cell in the outlet column, averaged, plus `(W-1)`, divided by `W`. The value is approximately `1 + true tortuosity`, not tortuosity itself. | does not match its name |
| B4 | `B4_max_enclosed_fluid_area_ratio` | `max_enclosed_fluid_pocket_over_bbox_area` | bbox | Largest connected fluid region that is **not** connected to the domain exterior, as a fraction of bounding-box area. Same bbox-area denominator caveat as A4. Measured constant `0.0` across the C-series canonical corpus (no dataset geometry in that corpus produces an enclosed pocket) -- see §4. | partially matches its name |
| C1 | `C1_frontal_area_density` | `windward_edge_pixel_count_per_bbox_area` | bbox | Count of windward-facing edge pixels (a length-like quantity) divided by bounding-box **area**, giving units of 1/px. The urban-climatology quantity this name borrows, frontal area density (lambda_f), is dimensionless; this is not. It also scales with image resolution (measured about x2.989 at 3x resolution, not the x3 a true areal density would show). | does not match its name |
| C2 | `C2_blockage_ratio` | `inlet_column_solid_fraction` | full | Solid fraction of the single x=0 column only -- unrelated to any obstacle's frontal projection. Numerically equals `1 - F1_inlet_fluid_fraction` (the unselected feature F1), so it is fully redundant with a feature that was not selected. Measured constant `0.0` on the C-series canonical corpus (buildings are confined to the interior of the domain in that generation convention) -- see §4. | does not match its name |
| C3 | `C3_shelter_index` | `fluid_fraction_downstream_of_first_solid` | bbox | Per row, find the first solid pixel's x position; count every fluid cell at or beyond that x as "sheltered." No distance decay, and no check for whether the flow re-exposes further downstream. | partially matches its name |
| D1 | `D1_mean_wall_distance_px` | `mean_edt_distance_to_nearest_solid_px` | bbox | Mean, over fluid cells, of the Euclidean distance transform to the nearest **building** cell. Neither the bounding-box crop edge nor the LBM domain wall counts as a wall, so the value runs large and depends on the bbox margin setting. | partially matches its name |
| D3 | `D3_hydraulic_diameter_est_px` | `shape_descriptor_4A_over_solid_perimeter_px` | bbox | `4 * fluid area / wetted perimeter`, where the perimeter counts only solid-fluid interface pixels (domain walls excluded). At sparse geometry the denominator approaches zero; measured on 493/670 (73.6%) of the C-series canonical corpus's nodes to exceed the full domain height (1080 px) -- not interpretable as a physical hydraulic diameter at that population. | partially matches its name |
| F3 | `F3_outlet_reachability` | `reachable_outlet_fluid_fraction` | full | Fraction of outlet-column (x = W-1) fluid cells that are BFS-reachable from the inlet. | matches its name |

`scope` is `bbox` (computed on the obstacle bounding box plus a 25% margin)
or `full` (computed on the entire LBM domain) -- this is a property of the
*computation*, defined in `features.py`, independent of the honesty
verdict.

## 3. Physics-claim status

Every one of the 14 dimensions has `claims_physics = False` in the source.
This is a blanket statement, not a per-row nuance: none of these 14 values
is asserted to equal the physical or urban-climatology quantity its
identifier borrows a name from, even where the `verdict` above is "matches
its name" (A3, A5, F3). "Matches its name" means the computation is a
faithful, literal implementation of what the name describes geometrically
(a count, a coefficient of variation, a reachable fraction) -- it is not a
claim that the quantity corresponds to, say, a real building's obstacle
count under some other definition, or that it predicts anything about
flow physics beyond being a geometric input to a learned model.

Identifiers were deliberately kept as-is rather than renamed to their
honest form, because they are already written into existing data files'
`feature_names` metadata and historical reports; renaming would desynchronise
those artifacts from the code without changing any computed value.

## 4. Degenerate dimensions

Three of the 14 dimensions -- B4, C2, F3 -- were measured to have exactly
zero sample standard deviation (`std <= 1e-6`) across a specific reference
corpus the source calls "C-series canonical" (670 nodes pooled across four
datasets): B4 and C2 constant at 0.0, F3 constant at 1.0. This is a
property of that corpus's mask-generation convention (buildings confined
to the domain interior, no enclosed courtyards, no full-height barrier
walls), not a property of the features themselves -- the source
demonstrates this with synthetic counter-examples (an enclosed courtyard
geometry gives B4 = 0.198; a building placed against the inlet gives C2 =
0.278). `GVEC_INERT_DIMS_C_SERIES` and `GVEC_EFFECTIVE_DIM_C_SERIES = 11`
name this specific corpus's degenerate set; a different training
population can have a different one (the source notes a 6-dimension
degenerate set when a smaller, 5-root-geometry dataset is used alone).

At training and deployment time, `TESCondNormalizer.normalize_g`
(`lbmnca.nca.tes.data.normalizer`) applies this mechanism generically: any
`G_vec` dimension whose *training-population* standard deviation is at or
below `1e-6` is z-scored to exactly `0` rather than divided by a
near-zero number, for whichever population the loaded statistics file
actually covers -- not hardcoded to the three C-series dimensions above.
A dimension that was constant throughout training receives no gradient at
its corresponding input weights, so injecting a real, non-constant value
for that dimension at deployment has no effect on the model's output; it
is multiplied by weights that were never updated.

## 5. Usage guidance

- Label figure axes and table columns with the `honest` name (§2), or with
  a description equivalent to it, not with the bare `code` identifier --
  the `code` identifier is what existing data files store, not what should
  be presented as a self-explanatory label.
- Do not describe any of the 14 dimensions as measuring the physical or
  urban-climatology quantity its name references. "C1 frontal area
  density" should be described as a windward-edge-pixel count normalised
  by bounding-box area, not as frontal area density lambda_f, even in
  passing.
- When citing degeneracy (B4/C2/F3 constant on the C-series corpus),
  attribute it to that corpus's generation convention, not to the features
  being inherently uninformative -- the synthetic counter-examples in §4
  show they do respond to geometry types simply absent from that
  particular training population.
- `D3`'s "not interpretable as a hydraulic diameter" caveat applies at the
  population level (73.6% of one corpus exceeding domain height); it is
  not a statement that the underlying `4A/P` computation is wrong, only
  that its physical-diameter interpretation breaks down when the wetted
  perimeter is small relative to fluid area.

## 6. API surface

`gvec_api.py` exposes:

- `compute_gvec(mask_path, *, white_is_solid=True)` -- from a mask PNG
  path. `white_is_solid=False` switches to the alternate convention
  (0 = building, 255 = fluid).
- `compute_gvec_from_mask(mask, *, solid_is_zero=True)` -- from an
  in-memory array, for callers that never touch disk (e.g. a live
  geometry edit). `solid_is_zero=True` expects 0 = solid / nonzero =
  fluid and internally flips it to the 1 = solid convention `features.py`
  expects.
- `compute_gvec_batch(mask_paths, ...)` -- batch wrapper; a mask that
  fails to compute (invalid geometry -- e.g. all-solid or all-fluid) gets
  a row of `NaN` rather than aborting the batch.
- `find_degenerate_dims(g_vecs, std_threshold=1e-6)` -- the same
  degeneracy test described in §4, exposed for computing it against any
  population of raw `G_vec` samples, not only the C-series reference.

All four return raw (non-normalised) `[..., 14]` `float32` arrays in
`GVEC_ACTIVE_INDICES` order; normalisation is a separate step
(`TESCondNormalizer`, §4).
