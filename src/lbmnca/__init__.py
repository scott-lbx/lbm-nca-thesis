"""Neural cellular automata surrogate for steady-state urban wind flow.

Subpackages
-----------
``lbmnca.morphospace``
    Procedural building-layout generation, the 14-dimensional geometric
    descriptor ``G_vec``, and the PCA design space used to sample cases.
``lbmnca.lbm``
    GPU lattice-Boltzmann solver (D2Q9, MRT collision, Smagorinsky LES) that
    produces the ground-truth dataset, plus HDF5-to-Zarr conversion.
``lbmnca.nca``
    The TES-NCA surrogate: model assembly, losses, training and evaluation.
``lbmnca.bridge``
    WebSocket inference server and renderer for real-time use inside
    Rhino/Grasshopper.

Paths to data and checkpoints are resolved by :mod:`lbmnca.paths`.
"""

from __future__ import annotations

__version__ = "1.0.0"
