"""convergence_overlay — Taichi kernel tinting active (unsettled) cells.

Renders the convergence front: cells still actively updating (convergence ~1)
get a faint cyan glow over the base colormap; settled cells (convergence ~0)
show the field cleanly. Pure GPU, runs after colorize, before overlay_buildings.
"""

import taichi as ti


@ti.kernel
def overlay_convergence(
    convergence: ti.template(),
    mask: ti.template(),
    rgb: ti.template(),
    strength: ti.f32,
):
    """Blend a cyan tint into fluid cells proportional to activity (convergence)."""
    for i, j in rgb:
        if mask[i, j] > 0.5:
            a = ti.min(ti.max(convergence[i, j], 0.0), 1.0) * strength
            c = rgb[i, j]
            tint = ti.Vector([0.2, 0.9, 1.0])
            rgb[i, j] = c * (1.0 - a) + tint * a
