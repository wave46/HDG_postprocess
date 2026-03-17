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

## Pointwise access

Pointwise access is grouped semantically:

```python
n = solution.pointwise.plasma.n(r, z)
ti = solution.pointwise.plasma.ti(r, z)
grad_ti = solution.pointwise.gradients.ti(r, z, "x")
btor = solution.pointwise.fields.magnetic_field(r, z, "theta")
q_loss = solution.pointwise.sources.Q_loss_total(r, z)
```

## Common workflows

### Sample a line

```python
solution.sample.define_interpolators()
line = solution.sample.line(r_line, z_line, ["n", "te", "ti", "M"])
```

### Compute power balance

Power balance usually needs atomic and neutral settings first, just as in the legacy workflow:

```python
power = solution.analysis.power_balance()
wall = solution.analysis.boundary_summary()
```

### Access Gauss and boundary views

```python
gauss_cons = solution.fields.conservative(view="gauss")
boundary_cons = solution.fields.conservative(view="boundary")
boundary_gauss_cons = solution.fields.conservative(view="boundary_gauss")
```

## Next reading

- [Mesh Quickstart](/home/ikudashev/Documents/Github/HDG_postprocess/docs/tutorials/mesh_quickstart.md)
- [Migrating Solution API](/home/ikudashev/Documents/Github/HDG_postprocess/docs/tutorials/migrating_solution_api.md)
- [migration_notes.md](/home/ikudashev/Documents/Github/HDG_postprocess/docs/refactor/migration_notes.md)
- demos in [demos](/home/ikudashev/Documents/Github/HDG_postprocess/demos)
