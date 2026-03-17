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
wall["ti"]
wall["n"]
wall["b_n"]
```

These arrays are already ordered along the assembled wall path, so they are ready for direct profile plotting.

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

## Compare with power balance

`power_balance()` and `boundary_summary()` are related, but they answer different questions:

- `power_balance()` integrates global volumetric gains and losses
- `boundary_summary()` builds an ordered wall profile and the boundary loss terms associated with it

That is why they are documented separately.

## Related docs

- [Power Balance](power_balance.md)
- [Setup Helpers](setup_helpers.md)
- demo notebook [hdg_solution_boundary_plots.ipynb](../../demos/hdg_solution_boundary_plots.ipynb)
