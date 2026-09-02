# Examples

Small runnable pieces of the pipeline. Start here to confirm an installation
works before committing GPU time to anything larger.

## `minimal_gvec/` — geometry and the descriptor

```bash
python examples/minimal_gvec/run.py
```

Generates a few building layouts, rasterises them, and prints the 14-dimensional
`G_vec` for each, writing the masks and a `gvec.csv` into
`examples/minimal_gvec/output/`.

Needs no GPU, no dataset, and no trained weights — only the `morphospace`
install. If this runs, the geometry half of the pipeline is working.

What each dimension actually measures, and which names are misleading, is in
[../docs/GVEC_SPEC.md](../docs/GVEC_SPEC.md).

> **Mask polarity.** Two conventions coexist: PNG files use `0` for solid, while
> in-memory boolean arrays use `True` for solid. `compute_gvec_from_mask` takes
> a `solid_is_zero` flag for exactly this reason. Getting it wrong is silent —
> the descriptor comes back full of plausible numbers that describe the
> background as one enormous building. The example shows the correct call.

## Running the model

```bash
python scripts/run_inference.py
```

Builds the network from `configs/nca/main.yaml`, loads
`checkpoints/main_v4.pth`, and reports the parameter count. This works without a
dataset, because the loader goes to the model factory rather than through the
training entry point.

## A single simulated case

The solver half needs a GPU. The smoke procedure in
[../docs/DATA_GENERATION.md](../docs/DATA_GENERATION.md) takes a handful of cases
from layout generation through simulation to the training format, which is the
right way to check that half before starting a full dataset run.
