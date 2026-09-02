# Contributing

Thanks for your interest. This repository accompanies a master's thesis, so its
scope is fixed: it exists to make the reported results reproducible. Bug reports,
portability fixes and documentation improvements are very welcome. Large feature
additions are likely out of scope.

## Development setup

```bash
git clone <this repository>
cd lbm-nca-thesis
python -m venv .venv
# Windows
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"
# POSIX
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

PyTorch and Taichi need attention: see the installation section of the README.
Neither is pinned to a CUDA build in `pyproject.toml`, because the right wheel
depends on your driver.

## Running the tests

```bash
pytest -m "not gpu"        # CPU only, no dataset needed. This is what CI runs.
pytest -m gpu              # needs a CUDA device and a regenerated dataset
```

Tests carrying the `gpu` marker require both a CUDA device and a locally
regenerated dataset, so they cannot run in CI.

## Style

- `ruff check .` and `ruff format .` must pass.
- Configuration is read with bracket access (`config["key"]`), never with a
  silent default. A missing key must raise, so that an experiment can never
  quietly run with parameters the configuration file does not contain.
- Public functions carry Google-style docstrings and type hints.
- Existing technical comments are written in Traditional Chinese. Keep the
  language of the file you are editing; do not translate surrounding comments
  as a side effect of an unrelated change.

## Reporting a bug

Please include the operating system, Python version, GPU and driver version, the
exact command, and the full traceback. If it concerns numerical results, say
which checkpoint and which configuration file you used.

## Scope note on results

Numbers reported in the thesis were produced with the checkpoint in
`checkpoints/` and the configuration in `configs/`. A pull request that changes
training behaviour will change those numbers, so it needs to state which results
it invalidates.
