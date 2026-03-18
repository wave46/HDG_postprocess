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

These accessors answer different questions:
- `metadata.p_order` describes the polynomial order of the stored elements.
- `metadata.extent` gives the mesh bounding box in physical coordinates.
- `global_state.vertices` and `global_state.connectivity` are the recombined full-mesh arrays.
- `derived_geometry.connectivity_big` is the plotting / triangulation connectivity built from the higher-order elements.

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

`mesh.assembly.full()` recombines partitioned mesh data into the global representation, `geometry.connectivity_big` prepares the dense plotting triangulation, `geometry.element_locator` gives the callable point-to-element lookup, and `adjacent_elements(...)` is a convenient local-topology query.

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

These calls order and group the raw boundary faces first, then build the boundary quadrature geometry that later wall summaries and sheath-loss evaluations use.

## Plotting

Plotting stays grouped under `mesh.plot`:

```python
mesh.plot.raw()
mesh.plot.full()
mesh.plot.boundary(raw_boundary_info)
```

Use `raw()` to inspect partition-local input data, `full()` for the recombined mesh, and `boundary(...)` when you want to visualize the ordered exterior boundary that the solution-side boundary workflows consume.

## Next reading

- [Solution Quickstart](solution_quickstart.md)
- [Migrating Mesh API](migrating_mesh_api.md)
- demo notebook [hdg_mesh.ipynb](../../demos/hdg_mesh.ipynb)
