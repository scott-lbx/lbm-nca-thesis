# Architecture

What the released model is conditioned on, how fields are normalised, and which
update path actually executes. Where a mechanism is present in the code but
disabled by the released configuration, this document says so rather than
describing it as a feature.

Configuration of record: `configs/nca/main.yaml`, weights `checkpoints/main_v4.pth`.

---

## 1. Spatial resolution

Three resolutions appear along the pipeline and are easy to confuse.

| Stage | Size | Meaning |
| --- | --- | --- |
| Solver domain | 2560 × 1440 | The lattice the LBM solver runs on. Width = 1920 mask + 160 inlet + 480 outlet; height = 1080 mask + 180 + 180 buffer. |
| ROI (mask) | 1920 × 1080 | The region occupied by the building layout, before downsampling. |
| Stored / inference | 640 × 360 | After 3× area downsampling. This is what lands in HDF5 and Zarr, and the resolution the surrogate operates at. |

## 2. Conditioning

### G_vec — 14 geometric scalars

Computed from the binary mask alone by
`lbmnca.morphospace.physics_proxy.gvec_api`. It selects 14 of the 22 features
defined in `features.py`:

```
A1 min_inter_obstacle_gap_px      A2 mean_inter_obstacle_gap_px
A3 obstacle_count                 A4 mean_obstacle_area_ratio
A5 obstacle_size_cv               B1 throat_width_px
B2 tortuosity                     B4 max_enclosed_fluid_area_ratio
C1 frontal_area_density           C2 blockage_ratio
C3 shelter_index                  D1 mean_wall_distance_px
D3 hydraulic_diameter_est_px      F3 outlet_reachability
```

Features that depend on the physical setup rather than on geometry alone
(effective Reynolds number, blockage-corrected velocity) are deliberately
excluded, so `G_vec` describes the layout and nothing else.

Several of these names borrow fluid-mechanics vocabulary for quantities that are
computed geometrically and do not equal the physical quantity the name suggests.
`docs/GVEC_SPEC.md` gives the honest per-dimension description; read it before
attributing physical meaning to any dimension.

### P_vec — 3 physical parameters

`[Re, U_inlet_lu, L_char_px]`, read directly from the HDF5 attributes written by
the solver.

### The conditioning vector

```
cond_dim = 17 = G_vec_norm (14) + P_vec_norm (3)
```

Assembled by `lbmnca.nca.tes.training.data_prep.condition_builder`. All released
configurations use `cond_dim: 17` and `state_dim: 21`.

**The conditioning vector carries no time information.** An earlier generation
used `cond_dim: 19`, where the two extra dimensions were the time step and the
cumulative time; both were constants in practice because the time-conditioning
path was never enabled. The released generation removes them, so 17 is a literal
count rather than an effective one. Do not describe the conditioning as
time-aware.

## 3. State

```
state_dim = 21 = static (4) + physics (9) + hidden (8)
```

**Static (4):** `[SDF_tanh, upstream_z, X01, Y01]`. Computed once per geometry.
During training these are read from precomputed Zarr tensors; at deployment
`lbmnca.bridge.core.static_builder` builds them on the fly from the mask
(Taichi jump-flooding for the distance field, with a SciPy EDT fallback).

**Physics (9):** the D2Q9 MRT moments in solver-native order,

```
[rho, e, eps, jx, qx, jy, qy, pxx, pxy]
```

so momentum sits at channels 3 and 5. This order is preserved end to end and
never permuted. Velocity, vorticity and pressure are derived quantities
(`vel = j / rho`, `p = rho / 3`) and **must be computed after denormalisation**.

**Hidden (8):** free channels, updated additively alongside physics and zeroed
inside solids.

## 4. Normalisation

Frozen constants, applied identically during training and deployment. They are
not recomputed from whatever data is at hand.

```
forward:  out = tanh((raw - M_REF) / S)
inverse:  raw = atanh(clip(out, ±0.9999)) * S + M_REF
```

`M_REF = [rho0, -2*rho0, rho0, 0, 0, 0, 0, 0, 0]` with `rho0 = 1`, derived from
the D2Q9 MRT rest equilibrium. `S` combines a per-channel baseline scale with a
group-shared rescale, so that direction is preserved within each group:

| Group | Channels | Sharing |
| --- | --- | --- |
| density | 0 | independent |
| energy | 1, 2 | shared |
| momentum | 3, 4, 5, 6 | shared, preserving flow direction |
| stress | 7, 8 | shared, preserving tensor orientation |

The signed distance field uses `tanh(SDF / (L_char_px / 3))`, per case, with zero
at the boundary. Position channels are domain-relative in `[0, 1]`.

`G_vec` and `P_vec` are stored raw and z-scored at runtime from the statistics
in `checkpoints/norm/`.

## 5. The update rule that actually runs

Each forward step: **multi-receptor perception → Markov cell update with a
stream-function momentum head → coarse/fine multiscale accumulation.** The
semi-Lagrangian transport stage is skipped throughout.

**Perception.** Five receptors in parallel: three fixed lattice-Boltzmann prior
kernels (topological, streaming, moment-transfer), a 48-channel learnable
convolution, and an 8-channel hormone field. The 101 combined channels are fused
down to 16 by a 1×1 convolution. This replaces the plain 3×3 perception of a
standard neural cellular automaton.

**Stream-function momentum head.** The projection head does not emit `jx` and
`jy` directly. It emits a scalar stream function ψ, which is low-pass filtered
with a fixed Gaussian (σ = 0.8) and then differentiated:

```
(jx, jy) = curl(ψ) = (∂ψ/∂y, -∂ψ/∂x)
```

with fixed, non-learnable central-difference kernels.

The motivation is that for a steady mean field `∂ρ/∂t ≈ 0`, so continuity
reduces to `∇·j = 0`, making divergence-free momentum the correct target. What
the construction actually guarantees is narrower:

- It guarantees the *increment* is divergence-free, in the fluid interior, under
  the same stencil used to take the curl. Measured at least three pixels from any
  boundary, `div(curl ψ)` is at machine precision.
- It does **not** remove divergence already present. A curl is the opposite of a
  Hodge projection: a projection removes existing divergence, a curl only
  promises not to add any. So `div(j_{n+1}) = div(j_n)`, and divergence carried
  in by a warm-start initial condition cannot be removed by the architecture.
- It does **not** hold at solid interfaces or domain boundaries. The write-back
  is `mask ∘ curl`, and `div(m·u) = m·div(u) + u·∇m`; the second term is nonzero
  wherever the mask changes.

The linear superposition of the coarse and fine branches preserves only the
increment property, not `div(j) = 0`. Do not restate this as "divergence-free by
construction".

> The low-pass filter has no train/deploy branch: σ must equal the training
> value of 0.8 at evaluation and deployment. Evaluating with a different σ is a
> configuration mismatch and invalidates the reading.

**Multi-edge chain training.** With probability 0.5, a training sample is a
two-stage chain across geometries, so the model is trained directly on the
"converged field of one layout becomes the warm start of the next" transition
rather than only on single hops.

**Pool-based self-repair.** A pool of past states is periodically reseeded,
corrupted with noise, and damaged with rectangular cutouts, so the update rule
learns to recover from a degraded field rather than only from clean ones.

## 6. Present in the code, inactive in the released configuration

Do not describe any of these as features of the released model.

| Mechanism | Why it does not run |
| --- | --- |
| Semi-Lagrangian transport warp | `transport.skip: true`; the warp encoder and displacement head are never constructed, and the streaming function returns early |
| Operator gating (`gate_stream`, `alpha_vel`) | The parameters exist in the checkpoint, but the only call site sits behind the skipped transport stage, so the forward pass never reaches them |
| Operator CREATE / PRIOR paths | `prior_mode: off` and `use_create: false` make them no-ops |
| Hand-off dilation reset | The dilation radius defaults to 0, so the reset degenerates to clearing the new geometry's solids |
| Stochastic asynchronous update | Rate defaults to 1.0, making the update fully synchronous |
| Kinetic-energy rescaling | Disabled in the configuration |

The architecture is registered under the name
`operator_gated_streamcollide`, but as the table shows, both the operator gating
and the streaming stage are inactive. The name records the lineage of the class
hierarchy, not the behaviour of the released model.

### Wired up, but with no measured gradient contribution

A Jacobian contraction regulariser is called every second step and costs a second
forward pass, but on the released weights its two-sided hinge evaluated to
exactly zero at every logged sample point, so its gradient contribution was zero
wherever it was observed. Its lower bound is switched off by configuration
(`jac_gamma_low: 0.0`), and the upper bound was never binding. Logging samples
only a small fraction of the steps on which it ran, so "zero at every logged
point" is an empirical statement about a sparse sample, not a proof. Do not
credit this term with suppressing long-rollout divergence without re-measuring.

## 7. Convergence criterion

Both the solver and the deployed surrogate stop on a two-window kinetic-energy
test: the most recent `2 × window` interval is split in half and the two means
are compared. This is robust to vortex shedding oscillation, which a single-window
test is not. The `convergence_ctu` field recorded in the dataset is the output of
this criterion.

At deployment the same criterion gates the stopping behaviour. If it is
disabled, the operator does not settle: the field keeps evolving, developing
visible banding after roughly a hundred steps, while the kinetic-energy ratio
stays well below its threshold and therefore never signals a problem. Leave
automatic stopping on unless you are deliberately studying the free-running
operator.

## 8. Solver limitations that shift the ground truth

The surrogate was trained against fields produced by the included solver, so
these propagate into every reported number.

- **Two-dimensional.** D2Q9. Any claim about pedestrian wind comfort inherits the
  absence of vertical flow structure.
- **Smagorinsky LES.** A two-dimensional LES closure is a modelling choice, not a
  resolved simulation. The eddy-viscosity term is embedded in the collision
  kernel.
- **Reynolds-number convention.** `Re` is defined in lattice units from the
  characteristic length `L_char_px`, which is the windward transverse extent of
  the layout. This is a specific convention; comparing against literature values
  requires matching it rather than assuming it.
- **Numerical stability** is maintained by a Mach-number limit, a lower bound on
  the relaxation time, and a sponge layer at the outflow. Cases that violate the
  admissibility box are rejected by the physics gate rather than silently kept.

## 9. Deployment initial conditions

Three modes, chosen by `lbmnca.bridge.integration.adapters`:

1. **Cold.** Physics channels start at zero. The surrogate is a warm-start
   refiner, so from a cold start alone it settles well below the ground-truth
   velocity scale. This mode is not recommended.
2. **Warm hand-off.** The previous geometry's converged state is carried over,
   with the new geometry's solids cleared. This is the intended interactive mode
   and the one the design of the model targets.
3. **Injected initial condition.** An externally supplied field is used directly,
   which is how evaluation chains are seeded.

For cold starts the released deployment first runs a separate autoregressive
baseline (`checkpoints/base_modern.pth`) for a few steps to produce a plausible
initial condition, then hands it to the surrogate to refine.
