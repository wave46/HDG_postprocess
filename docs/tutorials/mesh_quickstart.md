# Mesh Quickstart

This quickstart shows the preferred modern way to work with `HDGmesh`.

## Load a mesh

For new code, prefer:

```python
from hdg_postprocess.api import load_mesh

mesh = load_mesh(mesh_path, mesh_base, n_partitions)
```

Legacy loading through `hdg_postprocess.formats.load_from_file` still works.

## Inspect the main containers

Mesh state is grouped into:

```python
mesh.raw
mesh.metadata
mesh.global_state
mesh.derived_geometry
mesh.boundary_state
```

Examples:

```python
mesh.metadata.p_order
mesh.metadata.extent
mesh.global_state.vertices
mesh.global_state.connectivity
mesh.derived_geometry.connectivity_big
```

## Use grouped facades

The main grouped mesh facades are:

- `mesh.assembly`
- `mesh.geometry`
- `mesh.boundary`
- `mesh.plot`

Examples:

```python
mesh.assembly.full()
connectivity_big = mesh.geometry.connectivity_big
locator = mesh.geometry.element_locator
adjacent = mesh.geometry.adjacent_elements(42)
```

## Derived geometry and lazy properties

These are lazily initialized and cached:

```python
mesh.geometry.connectivity_big
mesh.geometry.mask
mesh.geometry.element_locator
mesh.geometry.gauss_volumes
```

That means you can access them like properties without manually recomputing them each time.

## Boundary assembly

Boundary operations remain method-based because they need explicit inputs:

```python
boundary = mesh.assembly.boundary(raw_boundary_info)
boundary_gauss = mesh.assembly.boundary_gauss(boundaries, raw_boundary_info)
```

## Plotting

Plotting stays grouped under `mesh.plot`:

```python
mesh.plot.raw()
mesh.plot.full()
mesh.plot.boundary(raw_boundary_info)
```

## Next reading

- [Solution Quickstart](/home/ikudashev/Documents/Github/HDG_postprocess/docs/tutorials/solution_quickstart.md)
- [Migrating Mesh API](/home/ikudashev/Documents/Github/HDG_postprocess/docs/tutorials/migrating_mesh_api.md)
- demo notebook [hdg_mesh.ipynb](/home/ikudashev/Documents/Github/HDG_postprocess/demos/hdg_mesh.ipynb)
