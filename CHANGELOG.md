# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-02

First public release, corresponding to the state of the code that produced the
results reported in the thesis.

### Added

- `lbmnca.morphospace` — procedural building-layout generation, the
  14-dimensional geometric descriptor `G_vec`, and the PCA design space used to
  sample dataset cases.
- `lbmnca.lbm` — GPU lattice-Boltzmann solver (D2Q9, MRT collision, Smagorinsky
  LES), HDF5 trajectory writer, physics-validation gate, and the HDF5-to-Zarr
  conversion that produces the training format.
- `lbmnca.nca` — the neural cellular automaton surrogate: model assembly, loss terms, training
  loop, evaluation metrics, and a checkpoint loader that builds a model without
  touching the training dataset.
- `lbmnca.bridge` — WebSocket inference server, Taichi GGUI renderer, and a
  self-contained Grasshopper component for real-time inference in Rhino.
- `checkpoints/main_v4.pth` — the canonical trained surrogate, together with the
  cold-start initial-condition generator and the frozen normalisation statistics.
- Documentation covering the architecture, the data formats, the full dataset
  regeneration procedure, and the reproduction steps.
