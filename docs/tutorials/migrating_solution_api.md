# Migrating Solution API

This note maps common older solution access patterns to the current structured API.

## Loaders

Old:

```python
from hdg_postprocess.formats import load_from_file
solution = load_from_file.load_HDG_solution_from_file(...)
```

New:

```python
from hdg_postprocess.api import load_solution
solution = load_solution(...)
```

The legacy loader is still supported, but `load_solution()` is the preferred entry point for new code.

## State access

Old style relied more heavily on split flat properties and ad hoc accessors.

New preferred style:

```python
solution.views.simple.solution.physical
solution.views.glob.solution.conservative
solution.views.gauss.gradient.conservative
solution.summary.boundary.profile
solution.metadata.flags
```

## Common method migrations

Old:

```python
solution.init_phys_variables()
solution.define_interpolators()
solution.calculate_power_balance()
solution.calculate_boundary_summary()
solution.plot_overview()
```

New:

```python
solution.fields.initialize_physical()
solution.sample.define_interpolators()
solution.analysis.power_balance()
solution.analysis.boundary_summary()
solution.plot.overview()
```

## Pointwise migrations

Old top-level pointwise access:

```python
solution.n(r, z)
solution.ti(r, z)
solution.grad_ti(r, z, "x")
solution.Q_loss_total(r, z)
```

New grouped pointwise access:

```python
solution.pointwise.plasma.n(r, z)
solution.pointwise.plasma.ti(r, z)
solution.pointwise.gradients.ti(r, z, "x")
solution.pointwise.sources.Q_loss_total(r, z)
```

## Sampling migrations

Old:

```python
solution.calculate_variables_along_line(r_line, z_line, variables)
```

New:

```python
solution.sample.line(r_line, z_line, variables)
```

## Boundary and Gauss views

Prefer these explicit accessors:

```python
solution.fields.conservative(view="gauss")
solution.fields.conservative(view="boundary")
solution.fields.conservative(view="boundary_gauss")
solution.fields.physical(view="simple")
solution.fields.physical(view="full")
```

## Recommendation

For migrated code:

- use facades for operations
- use `views` and `summary` for stored state
- avoid reintroducing removed flat wrapper patterns

## Related docs

- [Solution Quickstart](/home/ikudashev/Documents/Github/HDG_postprocess/docs/tutorials/solution_quickstart.md)
- [Migration Notes](/home/ikudashev/Documents/Github/HDG_postprocess/docs/refactor/migration_notes.md)
