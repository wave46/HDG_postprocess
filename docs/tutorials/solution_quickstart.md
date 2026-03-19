# Solution Quickstart

This quickstart shows the preferred modern way to load and inspect an `HDGsolution`.

## Load a solution

For new code, prefer the convenience loader:

```python
from hdg_postprocess.api import load_solution

solution = load_solution(
    solution_path,
    solution_base,
    mesh_path,
    mesh_base,
    n_partitions,
)
```

Legacy loaders still work, but the structured API below is the recommended surface for new analysis code.

## Inspect the main containers

State is grouped into a few predictable places:

```python
solution.views
solution.summary
solution.metadata
solution.additional_parameters
solution.raw
```

Typical examples:

```python
full_cons = solution.views.glob.solution.conservative
simple_phys = solution.views.simple.solution.physical
boundary_profile = solution.summary.boundary.profile
flags = solution.metadata.flags
```

These examples show the main roles of the containers:
- `views.glob.solution.conservative` is the assembled full-mesh conservative solution.
- `views.simple.solution.physical` is the reduced simple-mesh physical view that is often easiest to inspect.
- `summary.boundary.profile` stores the last computed boundary-summary table.
- `metadata.flags` tells you which assembled and derived states are already available.

## Use grouped facades for behavior

The main grouped facades are:

- `solution.fields`
- `solution.analysis`
- `solution.sample`
- `solution.plot`
- `solution.assembly`
- `solution.equilibrium`
- `solution.sources`
- `solution.neutrals`
- `solution.turbulence`
- `solution.pointwise`

Examples:

```python
full_cons = solution.fields.conservative(view="full")
simple_phys = solution.fields.physical(view="simple")
power = solution.analysis.power_balance()
profile = solution.sample.line(r_line, z_line, ["n", "te", "ti"])
```

`solution.fields.*` returns assembled arrays, `analysis.power_balance()` evaluates the integrated source and sink terms, and `sample.line(...)` interpolates the requested variables along an arbitrary curve without exposing the low-level interpolator machinery.

## Pointwise access

Pointwise access is grouped semantically:

```python
n = solution.pointwise.plasma.n(r, z)
ti = solution.pointwise.plasma.ti(r, z)
grad_ti = solution.pointwise.gradients.ti(r, z, "x")
btor = solution.pointwise.fields.magnetic_field(r, z, "theta")
q_loss = solution.pointwise.sources.Q_loss_total(r, z)
```

These pointwise calls are intended for local diagnostics: plasma accessors return local state variables, gradient accessors project a requested derivative component, field accessors expose magnetic-equilibrium quantities, and source accessors evaluate local source or loss terms.

## Common workflows

### Sample a line

```python
solution.sample.define_interpolators()
line = solution.sample.line(r_line, z_line, ["n", "te", "ti", "M"])
```

Calling `define_interpolators()` once prepares the reusable interpolation cache; the following `line(...)` call then reuses that setup to evaluate several variables efficiently at the same coordinates.

### Compute power balance

Power balance usually needs atomic and neutral settings first, just as in the legacy workflow:

```python
from hdg_postprocess.api import configure_solution_setup

configure_solution_setup(
    solution,
    reference_element=reference_element_path,
    radiation_model="nitrogen_cooling",
    atomic_data_dir="path/to/atomic",
    neutral_diffusion=True,
)
power = solution.analysis.power_balance()
wall = solution.analysis.boundary_summary()
```

The helper loads the reference element and common atomic / neutral-diffusion presets in one place. The atomic data directory stays explicit, so demo-local data paths are not silently baked into the library API.

### Access Gauss and boundary views

```python
gauss_cons = solution.fields.conservative(view="gauss")
boundary_cons = solution.fields.conservative(view="boundary")
boundary_gauss_cons = solution.fields.conservative(view="boundary_gauss")
```

These views expose the same solution on different discretization layouts: volume quadrature points, ordered boundary faces, and boundary quadrature points.

## Next reading

- [Mesh Quickstart](mesh_quickstart.md)
- [Point Sampling](point_sampling.md)
- [Power Balance](power_balance.md)
- [Boundary Summary](boundary_summary.md)
- [Setup Helpers](setup_helpers.md)
- [Migrating Solution API](migrating_solution_api.md)
- [migration_notes.md](../refactor/migration_notes.md)
- modern demo notebook [01_solution_basics.ipynb](../../demos/modern_api/01_solution_basics.ipynb)
- compatibility demos in [demos/compatibility](../../demos/compatibility)
