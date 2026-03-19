# HDG_postprocess

A package to read and postprocess SOLEDGE-HDG solutions.

The library is used to postprocess conservative, adimensional SOLEDGE-HDG output into
physical plasma quantities on high-order unstructured meshes. Supported workflows include
loading legacy and newer solution layouts, recombining partitioned meshes and solutions,
sampling with HDG-aware interpolators, and evaluating wall and power-balance diagnostics.

## Supported solution layouts

The current loader supports the combinations exercised by the demo datasets:

- Legacy partitioned solution files with an external mesh.
- Legacy single-file solution files with an external mesh.
- Newer grouped single-file solutions with an embedded mesh.
- Models with `n`, `Gamma`, `Ti`, `Te`, neutrals, and optional `k`.

These scenarios are locked down by baseline regression tests under [tests/baselines](tests/baselines).

## Compatibility status

The refactor is intentionally staged.

- Legacy loaders and compatibility paths remain supported for the workflows still covered by the regression suite and maintained demos.
- The modern API now returns the structured `HDGsolution` object directly.
- Numerical behavior is protected by regression baselines for loading, mesh recombination, physical variables, interpolated samples, sampled profiles, power balance, and boundary summaries.

The demo compatibility contract is recorded in [compatibility_contract.md](docs/refactor/compatibility_contract.md).

## Tutorials

For a lighter entry point than the full demo notebooks, start with:

- [Installation](docs/tutorials/installation.md)
- [Solution Quickstart](docs/tutorials/solution_quickstart.md)
- [Mesh Quickstart](docs/tutorials/mesh_quickstart.md)
- [Migrating Solution API](docs/tutorials/migrating_solution_api.md)
- [Migrating Mesh API](docs/tutorials/migrating_mesh_api.md)
- [Point Sampling](docs/tutorials/point_sampling.md)
- [Power Balance](docs/tutorials/power_balance.md)
- [Boundary Summary](docs/tutorials/boundary_summary.md)
- [Setup Helpers](docs/tutorials/setup_helpers.md)
- [IMAS Export](docs/tutorials/imas_export.md)

For canonical modern runnable examples, start with [demos/modern_api](demos/modern_api).

The longer legacy-compatible runnable examples remain in [demos/compatibility](demos/compatibility).

## Quick installation

For a fresh local environment, the simplest tested path is:

```bash
micromamba create -n hdg-postprocess-py312 -c conda-forge python=3.12 pip numpy scipy matplotlib cython silx pytest
micromamba activate hdg-postprocess-py312
pip install -e .
```

Then run a quick check:

```bash
python -m pytest tests/test_modern_api.py tests/test_mesh_ops.py -q
```

For the fuller local installation flow, see [Installation](docs/tutorials/installation.md).

## Setup data

Some workflows depend on local auxiliary data such as reference-element files and atomic tables. Those assets are not required for the core package layout, and they can be shared on request when you need to run the full demo or benchmark workflows. See [Setup Helpers](docs/tutorials/setup_helpers.md) for the expected local layout and setup patterns.

## Modern additive API

The legacy API remains supported:

```python
from hdg_postprocess.formats import load_from_file

solution = load_from_file.load_HDG_solution_from_file(...)
```

A modern convenience loader is also available for new code:

```python
from hdg_postprocess.api import load_solution

solution = load_solution(...)
full_cons = solution.fields.conservative(view="full")
simple_phys = solution.fields.physical(view="simple")
profile = solution.sample.line(r_line, z_line, ["n", "te", "ti"])
power = solution.analysis.power_balance()
```

For direct data access, prefer the structured container API instead of the long legacy split-property names:

```python
simple_phys = solution.views.simple.solution.physical
glob_cons = solution.views.glob.solution.conservative
gauss_grad = solution.views.gauss.gradient.conservative
boundary_profile = solution.summary.boundary.profile
```

The preferred user-facing shape is the structured `views` / `summary` access pattern together with the grouped facades on `HDGsolution`.

For pointwise sampling, the API is now also grouped semantically:

```python
n = solution.pointwise.plasma.n(r, z)
grad_ti = solution.pointwise.gradients.ti(r, z, "x")
btor = solution.pointwise.fields.magnetic_field(r, z, "theta")
q_loss = solution.pointwise.sources.Q_loss_total(r, z)
```

## Legacy API

Existing notebooks and scripts can continue using the long-lived loader entrypoints:

```python
from hdg_postprocess.formats import load_from_file

mesh = load_from_file.load_HDG_mesh_from_file(...)
solution = load_from_file.load_HDG_solution_from_file(...)
```

This remains the compatibility baseline during the refactor.

Examples of compatibility-only legacy access that new code should avoid:

```python
solution.views.boundary_gauss.gradient.conservative
```

These aliases are still kept for demo and notebook compatibility, but new code should use the structured container API instead.

## Dependencies

Current practical runtime dependencies include:

- `numpy`
- `scipy`
- `matplotlib`
- `silx` for HDF5 dictionary loading in the current file readers

`raysect` is no longer required for the production interpolation path. It is still useful as an optional comparison baseline in the locator benchmark tooling, and the transition history is documented in [locator_decision.md](docs/refactor/locator_decision.md).

## Python support

The current validated environments are:

| Python | Full `pytest tests -q` | Notes |
|---|---:|---|
| 3.8 | passed | legacy working environment |
| 3.10 | passed | good fresh-environment target |
| 3.11 | passed | slightly faster than 3.8/3.10 |
| 3.12 | passed | fastest among the tested versions |
| 3.13 | passed | works, with no extra code changes beyond the compatibility fixes already in this branch |

The package metadata still declares `requires-python = ">=3.8"`. For fresh installations, Python `3.11` or `3.12` is currently the most practical choice.

## Refactor status

The current branch has completed the safety-foundation phase:

- regression baselines generated from demo datasets
- automated regression tests for legacy and newer formats
- internal I/O normalization layer
- partial split of mesh and solution internals
- structured public API on `HDGsolution`

The most disruptive public-API cleanup is already done:

- grouped facades are the preferred public API
- structured containers are the preferred state surface
- the old flat wrapper layer has been removed

The main remaining refactor targets are:

- broader user documentation and tutor-style examples
- final polish of internal helper layers where needed
