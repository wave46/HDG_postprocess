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

These scenarios are locked down by baseline regression tests under [tests/baselines](/home/ikudashev/Documents/Github/HDG_postprocess/tests/baselines).

## Compatibility status

The refactor is intentionally staged.

- The legacy API remains supported for all workflows covered by the demo notebooks.
- The modern API now returns the structured `HDGsolution` object directly.
- Numerical behavior is protected by regression baselines for loading, mesh recombination, physical variables, interpolated samples, sampled profiles, power balance, and boundary summaries.

The demo compatibility contract is recorded in [compatibility_contract.md](/home/ikudashev/Documents/Github/HDG_postprocess/docs/refactor/compatibility_contract.md).

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

`raysect` is no longer required for the production interpolation path. It is still useful as an optional comparison baseline in the locator benchmark tooling, and the transition history is documented in [locator_decision.md](/home/ikudashev/Documents/Github/HDG_postprocess/docs/refactor/locator_decision.md).

## Refactor status

The current branch has completed the safety-foundation phase:

- regression baselines generated from demo datasets
- automated regression tests for legacy and newer formats
- internal I/O normalization layer
- partial split of mesh and solution internals
- structured public API on `HDGsolution`

The most disruptive `HDGsolution` cleanup is already done:

- structured state now lives in the dedicated `solution_api` package
- grouped facades are the preferred public API
- the old flat wrapper layer has been removed

The main remaining refactor targets are:

- `HDG_mesh.py`
- second-pass cleanup of `core/solution`
- second-pass cleanup of `routines`
- broader user documentation and tutor-style examples
