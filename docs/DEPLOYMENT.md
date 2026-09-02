# Deployment

Running the trained surrogate live, so that a layout edited in Rhino/Grasshopper
or drawn in the browser is solved while you work.

For step-by-step operating instructions, including the one-click launchers and
the full Grasshopper component setup, see [../demo/README.md](../demo/README.md).
This document covers the interface: what crosses the wire, and what the model
expects on each side of it.

---

## What runs where

One Python process plays three roles at once:

- an asyncio WebSocket server on port 8765, receiving JSON commands;
- PyTorch inference on the GPU;
- a Taichi GGUI window that renders the field.

They are joined by a thread-safe command queue and a zero-copy handoff of the
CUDA tensor to the renderer.

```
browser page  ──┐
                ├── WebSocket JSON ──► command queue ──► static channels + G_vec
Grasshopper   ──┘                                              │
                                                               ▼
                                                     surrogate rollout
                                                               │
                                                    zero-copy (same device)
                                                               ▼
                                                     Taichi GGUI window
```

**The flow field never travels back over the WebSocket.** Both clients receive
only an acknowledgement string. A browser sandbox cannot take a CUDA or Vulkan
handle, so the field is drawn in its own window rather than in the page. This is
a property of the architecture, not a missing feature.

## The wire protocol

An `analyze` command carries the geometry and the conditions:

| Field | Meaning |
| --- | --- |
| `polygons` | Building outlines. The browser sends rectangles in simulation pixels; Grasshopper sends arbitrary polygons in world coordinates. The server detects which dialect it received and normalises both. |
| `wind_angle` | Inflow direction, derived from the boundary rectangle's long axis |
| `wind_level` | 1, 2 or 3, mapping to Reynolds numbers 150, 300 and 500 |
| `mode` | `fresh` for a cold start, `warm` to continue from the current field |

Both clients may connect at once and send independently. They share one renderer
window, so whichever sent last owns the picture.

## What the model requires of the geometry

The layout is rasterised into a 640 × 360 mask, from which the static channels
(signed distance, upstream obstruction, position) and the 14-dimensional `G_vec`
are computed on the fly.

Two limits come from the training distribution. Outside them the model still
produces a field, but it is extrapolating and the component says so:

| Quantity | Trained range |
| --- | --- |
| `L_char` — windward transverse extent, in 1920 × 1080 simulation pixels | 32 to 128 px, i.e. 3 % to 12 % of the boundary's short side |
| `BCR` — total building area over boundary area | 0.01 to 0.30 |

The boundary is normalised to a unit square before rasterisation, so a boundary
that is not 16:9 stretches the geometry. Draw it 16:9.

**`G_vec` failure is fatal by design.** If the descriptor cannot be computed the
server raises rather than substituting a constant vector. A constant `G_vec`
would mean the flow no longer corresponds to the geometry on screen, with nothing
to indicate it. Do not switch the fallback on to make an error go away.

## Initial conditions

The intended interactive mode is the warm hand-off: the previous layout's
converged field is carried over, with the new layout's solids cleared, and the
surrogate refines from there. This is what the model was trained for and what
makes an edit cheaper than a fresh solve.

A cold start is weaker. The surrogate is a refiner, so from a zero field it
settles well below the true velocity scale. The released deployment therefore
runs a separate autoregressive baseline (`checkpoints/base_modern.pth`) for a few
steps first to produce a plausible field, then hands it over. That is why two
checkpoints ship rather than one.

## Stopping

The deployment stops on the same two-window kinetic-energy criterion the solver
uses. Leave automatic stopping enabled: with it off, the operator does not settle
at a fixed point but keeps evolving, developing visible banding after roughly a
hundred steps, and the kinetic-energy ratio does not rise enough to signal it.
Both the Grasshopper component and the renderer expose a switch for this; the
default is the safe one.

## Configuration

`configs/bridge/config.yaml`. Paths use the `${CHECKPOINT_ROOT}` and
`${REPO_ROOT}` placeholders resolved by `lbmnca.paths`, so the file needs no
editing on a fresh clone.

Without a checkpoint you can still start the server with `model.backend: dummy`,
which needs no weights and is useful for testing the geometry and rendering path
on a machine that has not pulled the LFS objects.
