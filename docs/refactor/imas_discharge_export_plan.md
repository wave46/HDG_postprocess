# IMAS Discharge Export Plan

This note sketches the next IMAS export mode after the current snapshot-oriented writer.

The goal is to export a full discharge represented by multiple SOLEDGE-HDG snapshot files into one IMAS netCDF-backed `DBEntry`.

## Target use case

- one physical discharge
- one `shot`
- one `run`
- multiple HDG snapshot files
- one IMAS `.nc` file

The current snapshot exporter already writes:

- `summary`
- `equilibrium`
- `plasma_profiles`

but only for one snapshot at a time.

The discharge exporter should extend that to time-resolved IDS content.

## Proposed API

Suggested public entrypoint:

```python
write_discharge_imas_netcdf(
    solutions,
    path,
    metadata,
    grid,
    *,
    include_summary=True,
    include_equilibrium=True,
    include_plasma_profiles=True,
    sort_by_time=True,
)
```

Where:

- `solutions`
  - ordered list of loaded `HDGsolution` snapshots
- `path`
  - output `.nc` file
- `metadata`
  - one shared `IMASExportMetadata` object for the discharge/run
- `grid`
  - one shared rectangular GGD grid

The time values for the snapshots should come from:

- `solution_time_seconds(solution)`

and should only be used when they are physically meaningful for the case.

## IDS shape

### summary

Keep a single run-level `summary` IDS.

Suggested content:

- description
- shot / run / occurrence
- run-level metadata from the first snapshot
- comment noting how many snapshots were exported
- optional list of exported times in `summary.code.parameters`

### equilibrium

Use one `equilibrium` IDS with:

- `equilibrium.time = [t0, t1, ...]`
- `equilibrium.grids_ggd`
  - shared once for the whole discharge
- `equilibrium.time_slice.resize(N)`
- `equilibrium.time_slice[i].time = times[i]`
- `equilibrium.time_slice[i].ggd[...]`
  - one GGD payload per snapshot

### plasma_profiles

Use one `plasma_profiles` IDS with:

- `plasma_profiles.time = [t0, t1, ...]`
- `plasma_profiles.grid_ggd`
  - shared once for the whole discharge
- `plasma_profiles.ggd.resize(N)`
- `plasma_profiles.ggd[i].time = times[i]`
- species-resolved fields in each `ggd[i]`

## Implementation notes

1. Build and validate one shared grid once.
2. Sort snapshots by time if requested.
3. Reuse the current snapshot field-mapping logic for each time index.
4. Keep the same node subset conventions as the current rectangular-GGD exporter.
5. Write the final IDSs once with `put()`.

The netCDF backend is better suited to writing the assembled IDSs once than to repeated slice-style appends.

## Success criterion

The first discharge-export milestone is:

- multiple HDG snapshot files
- one IMAS `.nc` file
- one `summary`
- one `equilibrium` with multiple `time_slice`
- one `plasma_profiles` with multiple `ggd`
- successful readback of at least one time index by a downstream user
