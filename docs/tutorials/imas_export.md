# IMAS Export

This note shows the compact export flow for the current IMAS adapter layer.

The first exporter writes three IDSs into one netCDF-backed `DBEntry`:

- `summary`
- `equilibrium`
- `plasma_profiles`

The current 2D representation is a rectangular cylindrical `(R, Z)` GGD mesh generated on the writer side.

When `equilibrium` and `plasma_profiles` are written together into the same `DBEntry`, the combined writers now store the explicit rectangular GGD topology in `plasma_profiles.grid_ggd` and let `equilibrium.grids_ggd` reference it through the IMAS `path` field. This avoids duplicating the same grid description twice in the same file.

## Current Export Contract

The current IMAS adapter already supports three practical patterns:

- one steady-state case -> one `.nc` file
- one bundled scan -> one `.nc` file with one IDS occurrence per simulation
- one full discharge -> one `.nc` file with time arrays and multiple GGD payloads

Two small implementation details are worth knowing up front:

- `RectangularGrid2D.from_solution_bounds(...)` now assembles the full mesh automatically when a legacy multi-partition case has not yet recombined its global mesh vertices
- unchanged meshes can now reuse the shared sample-interpolator geometry cache across separately loaded solutions, which helps repeated scan/discharge exports at the same `(R, Z)` sampling points
- when `equilibrium` and `plasma_profiles` are exported together, `equilibrium.grids_ggd` now references `plasma_profiles.grid_ggd` instead of storing a duplicate rectangular topology

## Recommended File Layout

For the current exporter, the recommended unit is:

- one simulation case or one simulation snapshot
- one netCDF-backed IMAS `DBEntry`
- one `.nc` file

Inside that one file, write:

- `summary`
- `equilibrium`
- `plasma_profiles`

This keeps each exported file conceptually clean:

- one steady-state simulation -> one file
- one puff scan -> one file, with one IDS occurrence per scan point
- one full discharge -> one file, with multiple time entries in `equilibrium` and `plasma_profiles`

For a puff scan, keep the same `shot`, keep `run=0`, and use one IDS occurrence per scan point.
The current project convention is to bundle the scan into one `.nc` file by using one IDS occurrence per simulation.
For a full discharge, keep one `shot`, one `run`, one `occurrence`, and export the ordered snapshot list into one time-resolved `DBEntry`.

To reduce redundant storage, the current plasma export fills only the authoritative fields for the single-ion SOLEDGE-HDG model:

- `electrons.density`
- `electrons.temperature`
- `ion[0].temperature`
- `ion[0].velocity.parallel`
- `neutral[0].density`

The redundant `ion[0].density`, `n_i_total`, and `t_i_average` fields are intentionally left empty and documented in `plasma_profiles.code.parameters`.
Likewise, `equilibrium.time_slice[i].ggd` no longer duplicates `R` and `Z` as data fields, because those coordinates are already defined by the referenced GGD topology.
If `Zeff` is spatially constant, it is exported through `plasma_profiles.global_quantities.z_eff_resistive` instead of as a 2D field.

## One Steady-State Solution

Use an explicit user-provided time for a single steady-state snapshot.
For steady runs, the solver `Current_time` is usually not the physical experiment time you want to expose in IMAS.

```python
from hdg_postprocess.api import load_reference_element, load_solution
from hdg_postprocess.imas_export import (
    IMASExportMetadata,
    RectangularGrid2D,
    SolutionSnapshotSource,
    write_discharge_imas_netcdf,
    write_imas_netcdf,
    write_imas_scan_case_netcdf,
)

solution = load_solution(
    "demos/data/solutions/power_balance_boundary_checks/",
    "solution_west_heating_cooling_factor",
    n_partitions=1,
)
solution.mesh.metadata.reference_element = load_reference_element(
    "demos/data/reference_elements/reference_triangle_P8.mat"
)

metadata = IMASExportMetadata(
    description="Single steady-state SOLEDGE-HDG export",
    shot=1,
    run=1,
    time=0.0,
    effective_energy_transfer=1.0,
    comment="Prototype IMAS export.",
    machine="WEST",
)

grid = RectangularGrid2D.from_solution_bounds(solution, dr=0.005, dz=0.005)

write_imas_netcdf(
    solution,
    "build/imas_single_case.nc",
    metadata,
    grid,
    file_mode="w",
)
```

For legacy multi-partition solutions, `RectangularGrid2D.from_solution_bounds(...)` now takes care of assembling the full mesh before reading global vertex bounds, so you do not need an extra mesh-recombination step just to build the export grid.

## Puff Scan for One Shot

For a puff scan, keep the same `shot`, keep `run=0`, and use one IDS occurrence per scan point.
Store the whole scan in one `.nc` file and set `occurrence` explicitly for each simulation.

```python
base_shot = 60527
scan_path = "build/puff_scan.nc"
for occurrence, solname in enumerate(
    [
        "solution_west_scan_puff_01",
        "solution_west_scan_puff_02",
        "solution_west_scan_puff_03",
    ]
):
    solution = load_solution(
        "path/to/scan/",
        solname,
        n_partitions=1,
    )
    solution.mesh.metadata.reference_element = load_reference_element(
        "demos/data/reference_elements/reference_triangle_P8.mat"
    )

    metadata = IMASExportMetadata(
        description=f"Puff scan case {occurrence}",
        shot=base_shot,
        run=0,
        occurrence=occurrence,
        time=0.0,
        effective_energy_transfer=1.0,
        machine="WEST",
    )

    grid = RectangularGrid2D.from_solution_bounds(solution, dr=0.005, dz=0.005)
    write_imas_scan_case_netcdf(
        solution,
        scan_path,
        metadata,
        grid,
        create=(occurrence == 0),
    )
```

When reading the scan back, use the same occurrence number across the IDSs:

```python
with imas.DBEntry("build/puff_scan.nc", "r") as entry:
    occurrence = 1
    summary = entry.get("summary", occurrence)
    equilibrium = entry.get("equilibrium", occurrence)
    plasma = entry.get("plasma_profiles", occurrence)
```

## Full Discharge in One DBEntry

For a full discharge, provide the ordered list of snapshot files and export them into one `.nc` file.
This mode expects one physically meaningful time value per snapshot.
You may pass either:

- one shared rectangular grid
- or one rectangular grid per snapshot if the mesh changes during the discharge

```python
from hdg_postprocess.imas_export import (
    IMASExportMetadata,
    RectangularGrid2D,
    SolutionSnapshotSource,
    write_discharge_imas_netcdf,
)

snapshot_names = [
    "solution_snapshot_0000",
    "solution_snapshot_0001",
    "solution_snapshot_0002",
]

snapshots = [
    SolutionSnapshotSource(
        solution_path="path/to/discharge/",
        solution_base=snapshot_name,
        n_partitions=1,
        reference_element="demos/data/reference_elements/reference_triangle_P8.mat",
    )
    for snapshot_name in snapshot_names
]

metadata = IMASExportMetadata(
    description="Full discharge export",
    shot=1,
    run=42,
    time=0.0,
    effective_energy_transfer=1.0,
    machine="WEST",
)

grids = []
for snapshot in snapshots:
    solution = snapshot.load()
    grids.append(RectangularGrid2D.from_solution_bounds(solution, dr=0.005, dz=0.005))
    del solution
write_discharge_imas_netcdf(
    snapshots,
    "build/full_discharge.nc",
    metadata,
    grids,
    file_mode="w",
    sample_workers=4,
)
```

By default, `write_discharge_imas_netcdf(...)` uses `solution_time_seconds` and expects snapshots to already be ordered in time.
If the stored solver time is not physically meaningful for the exported discharge, or if you need a stricter validity rule, pass your own `time_getter` callable instead.
When the input is given as `SolutionSnapshotSource` objects, `sample_workers` lets the exporter load snapshots and sample plasma/equilibrium fields in multiple Python processes before the final IMAS IDS assembly and `put()` phase.

## Performance Notes

The current exporter is correct and reasonably reusable, but it is not lightweight on large grids.

The main costs are:

- repeated snapshot loading
- evaluating plasma and equilibrium fields on every `(R, Z)` sample point
- building the explicit GGD node/edge/cell topology for the explicit grid entries
- the final `put()` calls, especially when validation is enabled

Recent improvements already in the current branch:

- equilibrium export now uses batched interpolator evaluation on the valid grid points instead of repeated single-point calls from the exporter loop
- unchanged meshes can reuse the shared sample-interpolator geometry cache across separately loaded solutions
- full-discharge export can stream snapshots from `SolutionSnapshotSource` instead of keeping all `HDGsolution` objects resident
- when `equilibrium` and `plasma_profiles` are exported together, `equilibrium` references `plasma_profiles.grid_ggd` instead of duplicating the same rectangular topology
- discharge export now shares one sampling pass between plasma and equilibrium and can parallelize that sampling stage with `sample_workers`

Still worth keeping in mind:

- the interpolator geometry cache is currently unbounded
- plasma export is still slower than equilibrium export because it samples more derived variables
- `IMAS_AL_DISABLE_VALIDATE=1` can reduce `put()` time noticeably in production runs, but only use it after one validated smoke test of the same workflow
- `sample_workers` controls the number of Python worker processes used for snapshot loading and sampling; it is independent of `OMP_NUM_THREADS`
- on cluster, start with something like `sample_workers=4` or `8` and benchmark before going much higher, because returns become smaller as worker count increases

## Inspecting the Written File

Avoid calling `print_tree()` on the whole GGD IDS when the grid is large.
Inspect the small branches instead.

```python
import imas
from imas.util import print_tree

with imas.DBEntry("build/imas_single_case.nc", "r") as entry:
    equilibrium = entry.get("equilibrium", 0)
    plasma = entry.get("plasma_profiles", 0)

print_tree(equilibrium.grids_ggd[0].grid[0].grid_subset[0], hide_empty_nodes=True)
print_tree(equilibrium.time_slice[0].ggd[0].psi[0], hide_empty_nodes=True)
print_tree(plasma.ggd[0].electrons.density[0], hide_empty_nodes=True)
```

## Cluster Batch Pattern

For cluster use, the usual pattern is:

1. prepare one Python environment with `imas-python` and `hdg_postprocess`
2. loop over solution folders or snapshot names
3. choose `shot` / `run` explicitly
4. write one netCDF file per steady case, per bundled scan, or per full discharge

That keeps the export script simple and makes it easy to transfer the resulting `.nc` files to other users.

For a large full discharge on cluster, a practical pattern is:

1. request enough CPUs from the scheduler
2. set `sample_workers` explicitly to the number of Python sampling processes you want to use
3. optionally set `IMAS_AL_DISABLE_VALIDATE=1` after one validated smoke test

`sample_workers` is not inferred from `OMP_NUM_THREADS`. The exporter uses Python worker processes for the sampling stage, so you should tune `sample_workers` directly in the export call to match the CPU allocation you want to use.
