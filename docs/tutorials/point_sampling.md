# Point Sampling

This guide shows the intended workflow for evaluating solution quantities at isolated points and along lines.

## When to use which API

Use `solution.sample.*` when you want several variables at the same coordinates and want the interpolator cache handled for you.

Use `solution.pointwise.*` when you want one specific local diagnostic and prefer a more semantic access path.

## Minimal setup

Point sampling needs a reference element before interpolators can be constructed:

```python
from hdg_postprocess.api import load_solution, load_reference_element

solution = load_solution(solution_path, solution_base, mesh_path, mesh_base, n_partitions)
solution.mesh.metadata.reference_element = load_reference_element(reference_element_path)
```

The reference element provides the shape functions and quadrature layout needed to evaluate fields inside the high-order elements.

## Sample one point

```python
solution.sample.define_interpolators()
point = solution.sample.point(r, z, ["n", "te", "ti"])
```

`define_interpolators()` builds the reusable interpolation objects once. `sample.point(...)` then returns a small dictionary keyed by the requested variable names.

## Sample a line

```python
solution.sample.define_interpolators()
profile = solution.sample.line(r_line, z_line, ["n", "te", "ti", "M"])
```

This is the preferred workflow when all variables are evaluated on the same set of coordinates. The current implementation walks point-by-point and reuses the cached geometric lookup across the requested variables.

Typical result layout:

```python
profile["n"]
profile["te"]
profile["ti"]
profile["M"]
```

Each entry has the same length as `r_line` and `z_line`.

## Use semantic pointwise accessors

```python
n = solution.pointwise.plasma.n(r, z)
ti = solution.pointwise.plasma.ti(r, z)
mach = solution.pointwise.plasma.mach(r, z)
grad_ti_x = solution.pointwise.gradients.ti(r, z, "x")
b_theta = solution.pointwise.fields.magnetic_field(r, z, "theta")
q_loss = solution.pointwise.sources.Q_loss_total(r, z)
```

These calls are useful when the physics meaning matters more than the raw variable list. They also make analysis scripts easier to scan later because plasma, gradients, fields, and source terms live in separate subfacades.

## Practical recommendation

- call `solution.sample.define_interpolators()` once before many repeated queries
- use `sample.line(...)` or `sample.point(...)` for grouped sampling workloads
- use `pointwise.*` for isolated semantic diagnostics

## Related docs

- [Solution Quickstart](solution_quickstart.md)
- [Power Balance](power_balance.md)
- [Setup Helpers](setup_helpers.md)
- demo notebook [hdg_solution_west.ipynb](../../demos/hdg_solution_west.ipynb)
