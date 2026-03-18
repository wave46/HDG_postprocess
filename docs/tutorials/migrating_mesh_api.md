# Migrating Mesh API

This note maps common older mesh access patterns to the current grouped API.

## Loaders

Old:

```python
from hdg_postprocess.formats import load_from_file
mesh = load_from_file.load_HDG_mesh_from_file(...)
```

New:

```python
from hdg_postprocess.api import load_mesh
mesh = load_mesh(...)
```

## State access

Old code often relied on many flat properties on `HDGmesh`.

New preferred state layout:

```python
mesh.raw
mesh.metadata
mesh.global_state
mesh.derived_geometry
mesh.boundary_state
```

This split is intentional: raw partition payload, metadata and flags, recombined global mesh arrays, cached derived geometry, and ordered boundary state now live in separate predictable places.

Examples:

```python
mesh.global_state.vertices
mesh.global_state.connectivity
mesh.derived_geometry.connectivity_big
mesh.boundary_state.normals_gauss
```

## Geometry and assembly migrations

Old:

```python
mesh.recombine_full_mesh()
mesh.recombine_full_boundary(raw_boundary_info)
mesh.calculate_gauss_boundary(boundaries, raw_boundary_info)
mesh.connectivity_big
mesh.element_number
mesh.volumes_gauss
```

New:

```python
mesh.assembly.full()
mesh.assembly.boundary(raw_boundary_info)
mesh.assembly.boundary_gauss(boundaries, raw_boundary_info)
mesh.geometry.connectivity_big
mesh.geometry.element_locator
mesh.geometry.gauss_volumes
```

The distinction is that `assembly.*` methods actively build or reorder mesh state, while `geometry.*` exposes cached derived products that can then be reused by sampling, plotting, and integration workflows.

## Plotting migrations

Old flat plotting methods were replaced by the grouped plotting facade:

```python
mesh.plot.raw()
mesh.plot.full()
mesh.plot.boundary(raw_boundary_info)
```

Plotting remains procedural because these calls produce figures, but it is now grouped behind one dedicated facade instead of being mixed with geometry-building methods.

## Recommendation

For migrated mesh code:

- use `assembly` for build/recombine workflows
- use `geometry` for cached derived geometry
- use containers for stored state
- avoid depending on removed flat cache-holder properties

## Related docs

- [Mesh Quickstart](mesh_quickstart.md)
- [locator_decision.md](../refactor/locator_decision.md)
