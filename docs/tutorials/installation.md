# Installation

This guide covers the tested local installation path for development and everyday use.

For now it focuses on a fresh local user-space environment. Cluster-specific notes can be added later on top of the same workflow.

## Recommended Python versions

The package metadata allows `>=3.8`, but the most practical fresh-install targets are:

- Python `3.11`
- Python `3.12`

Both were validated with the full regression suite.

## Recommended local workflow

If `micromamba` is not already installed, one simple user-space setup is:

```bash
mkdir -p ~/.local/bin
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~/.local/bin --strip-components=1 bin/micromamba
export MAMBA_ROOT_PREFIX="$HOME/.micromamba"
```

These are reasonable default locations for a local no-sudo installation:

- `~/.local/bin` for the `micromamba` executable
- `~/.micromamba` for environments and package caches

If `~/.local/bin` is already on your `PATH`, the `micromamba` command should work immediately. Otherwise, either:

- call it as `~/.local/bin/micromamba`, or
- add `~/.local/bin` to your shell `PATH`

For interactive use, it is also worth initializing shell support once:

```bash
~/.local/bin/micromamba shell init -s bash -r "$HOME/.micromamba"
```

Then restart the shell, or source your shell configuration, so `micromamba activate ...` works normally.

Create a fresh micromamba environment with the core runtime and build dependencies:

```bash
micromamba create -n hdg-postprocess-py312 -c conda-forge python=3.12 pip numpy scipy matplotlib cython silx pytest
micromamba activate hdg-postprocess-py312
```

Then install the package in editable mode from the repository root:

```bash
pip install -e .
```

Editable install is convenient while developing because changes in the Python sources are picked up directly from the working tree.

## Quick verification

Run a small smoke-test slice:

```bash
python -m pytest tests/test_modern_api.py tests/test_mesh_ops.py -q
```

If you want the full regression check:

```bash
python -m pytest tests -q
```

## Build notes

The package uses `pyproject.toml` for project metadata and build requirements, and `setup.py` for the explicit Cython extension build step.

That means source builds expect:

- `Cython`
- `numpy`
- a working C compiler toolchain

## Setup data

Some tutorials, demos, and benchmark workflows also need local auxiliary files such as:

- reference-element `.mat` files
- atomic `.npy` tables

Those are described in [Setup Helpers](setup_helpers.md).

## Related docs

- [Solution Quickstart](solution_quickstart.md)
- [Mesh Quickstart](mesh_quickstart.md)
- [Setup Helpers](setup_helpers.md)
