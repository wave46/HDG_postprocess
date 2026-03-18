# IMAS Export

This note shows the compact export flow for the current IMAS adapter layer.

The first exporter writes three IDSs into one netCDF-backed `DBEntry`:

- `summary`
- `equilibrium`
- `plasma_profiles`

The current 2D representation is a rectangular cylindrical `(R, Z)` GGD mesh generated on the writer side.

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
- one time-resolved snapshot -> one file

For a puff scan, keep the same `shot` and increment `run`.
The current project convention is to bundle the scan into one `.nc` file by using one IDS occurrence per scan point.
For full-discharge snapshots, you can still keep the same `shot` and use `run` or file naming to distinguish snapshots, depending on how you want to organize the dataset on disk.

## One Steady-State Solution

Use an explicit user-provided time for a single steady-state snapshot.
For steady runs, the solver `Current_time` is usually not the physical experiment time you want to expose in IMAS.

```python
from hdg_postprocess.api import load_reference_element, load_solution
from hdg_postprocess.imas_export import (
    IMASExportMetadata,
    RectangularGrid2D,
    write_imas_netcdf,
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

## Puff Scan for One Shot

For a puff scan, keep the same `shot` and change `run` for each scan point.
Store the whole scan in one `.nc` file and use one IDS occurrence per scan point.

```python
base_shot = 60527
scan_path = "build/puff_scan.nc"
for run_in_scan, solname in enumerate(
    [
        "solution_west_scan_puff_01",
        "solution_west_scan_puff_02",
        "solution_west_scan_puff_03",
    ],
    start=1,
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
        description=f"Puff scan case {run_in_scan}",
        shot=base_shot,
        run=run_in_scan,
        occurrence=run_in_scan - 1,
        time=0.0,
        effective_energy_transfer=1.0,
        machine="WEST",
    )

    grid = RectangularGrid2D.from_solution_bounds(solution, dr=0.005, dz=0.005)
    write_imas_netcdf(
        solution,
        scan_path,
        metadata,
        grid,
        file_mode="w" if run_in_scan == 1 else "a",
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

## Full Discharge or Time-Resolved Snapshots

For a time-dependent simulation, reuse the dimensionalized time stored in the solution only when that time is physically meaningful for the exported case.
In practice, this means non-steady moving-equilibrium / time-resolved snapshots rather than steady-state solutions.

```python
from hdg_postprocess.imas_export import solution_time_seconds

solution = load_solution(
    "path/to/discharge/",
    "solution_snapshot_0420",
    n_partitions=1,
)
solution.mesh.metadata.reference_element = load_reference_element(
    "demos/data/reference_elements/reference_triangle_P8.mat"
)

metadata = IMASExportMetadata(
    description="Full-discharge snapshot export",
    shot=1,
    run=42,
    time=solution_time_seconds(solution),
    effective_energy_transfer=1.0,
    machine="WEST",
)

grid = RectangularGrid2D.from_solution_bounds(solution, dr=0.005, dz=0.005)
write_imas_netcdf(solution, "build/discharge_snapshot_0420.nc", metadata, grid, file_mode="w")
```

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
4. write one netCDF file per exported case or per bundled scan

That keeps the export script simple and makes it easy to transfer the resulting `.nc` files to other users.
