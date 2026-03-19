# Boundary Summary

This guide focuses on the wall-profile workflow behind `solution.analysis.boundary_summary()`.

## What this workflow computes

`boundary_summary()` evaluates ordered boundary quantities such as:

- wall coordinates
- plasma state along the boundary
- normal magnetic-field projections
- integrated sheath-loss terms

The result is cached in `solution.summary.boundary.profile`.

## Minimal setup

Boundary summary needs the same reference-element and, in many cases, atomic / neutral setup that power-balance workflows use:

```python
from hdg_postprocess.api import configure_solution_setup, load_solution

solution = load_solution(solution_path, solution_base, mesh_path, mesh_base, n_partitions)

configure_solution_setup(
    solution,
    reference_element=reference_element_path,
    radiation_model="none",
    atomic_data_dir="path/to/atomic",
    neutral_diffusion=True,
)
```

The reference element is needed for the ordered boundary-Gauss representation. The extra atomic and neutral settings are required if your later wall analysis uses source-term-derived quantities.

## Compute the boundary summary

```python
wall = solution.analysis.boundary_summary()
```

Typical profile access:

```python
wall["r"]
wall["z"]
wall["te"]
wall["te_skeleton"]
wall["ti"]
wall["ti_skeleton"]
wall["n"]
wall["n_skeleton"]
wall["b_n"]
```

These arrays are already ordered along the assembled wall path, so they are ready for direct profile plotting.

In most wall-facing analyses, the `*_skeleton` variants are the more natural quantities to use because they come from the trace solution stored on the mesh skeleton. The non-skeleton values are usually very close, but they come from the element-interior conservative state evaluated on the boundary representation rather than from the trace variable itself.

## Identify a nearby boundary face

The mesh API now includes a small helper for locating the nearest assembled boundary face to a point:

```python
boundary_flags = tuple(np.unique(solution.raw.boundary_infos[0]["boundary_flags"]).tolist())
idx = solution.mesh.boundary.nearest_face_index(
    r_probe,
    z_probe,
    raw_boundary_info=solution.raw.boundary_infos[0],
    boundaries=boundary_flags,
)
```

This is useful when you want to mark or slice a particular wall region in a notebook without keeping a local helper function around.

## Related docs

- [Power Balance](power_balance.md)
- [Setup Helpers](setup_helpers.md)
- compatibility demo notebook [hdg_solution_boundary_plots.ipynb](../../demos/compatibility/hdg_solution_boundary_plots.ipynb)
