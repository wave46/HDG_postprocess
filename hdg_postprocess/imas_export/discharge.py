import gc
import json

import numpy as np

from .common import add_constant_zeff_metadata, extend_time_metadata
from .config import IMASExportMetadata, RectangularGrid2D, SolutionSnapshotSource
from .equilibrium import build_equilibrium_metadata, populate_equilibrium_timeslice, sample_equilibrium_fields
from .ggd_geometry import populate_grid_reference_ggd_entry, populate_rectangular_grid_ggd_entry
from .plasma_profiles import build_plasma_profiles_metadata, populate_plasma_ggd, sample_plasma_fields, set_constant_zeff
from .summary import build_summary_export_metadata, populate_summary_ids


def write_discharge(
    entry,
    solutions,
    metadata: IMASExportMetadata,
    grid,
    *,
    include_summary=True,
    include_equilibrium=True,
    include_plasma_profiles=True,
    sort_by_time=False,
    time_getter=None,
):
    if not solutions:
        raise ValueError("Full-discharge export requires at least one solution snapshot.")
    if time_getter is None:
        raise ValueError("A time_getter callable must be provided for full-discharge export.")

    grids = build_discharge_grids(len(solutions), grid)
    timed_snapshots = build_timed_snapshots(
        solutions,
        grids,
        sort_by_time=sort_by_time,
        time_getter=time_getter,
    )
    times = [time_value for time_value, _, _ in timed_snapshots]

    written = {}
    if include_summary:
        summary = build_discharge_summary_ids(timed_snapshots, metadata, times)
        entry.put(summary, metadata.occurrence)
        written["summary"] = build_written_ids_info(metadata, times)
        release_large_object(summary)

    grid_reference_paths = None
    if include_equilibrium and include_plasma_profiles:
        grid_reference_paths = build_plasma_grid_reference_paths(metadata.occurrence, len(timed_snapshots))

    if include_plasma_profiles:
        plasma = build_discharge_plasma_profiles_ids(timed_snapshots, metadata)
        entry.put(plasma, metadata.occurrence)
        written["plasma_profiles"] = build_written_ids_info(metadata, times)
        release_large_object(plasma)

    if include_equilibrium:
        equilibrium = build_discharge_equilibrium_ids(
            timed_snapshots,
            metadata,
            grid_reference_paths=grid_reference_paths,
        )
        entry.put(equilibrium, metadata.occurrence)
        written["equilibrium"] = build_written_ids_info(metadata, times)
        release_large_object(equilibrium)

    return written


def build_discharge_summary_ids(timed_snapshots, metadata, times):
    import imas

    _, first_snapshot, _ = timed_snapshots[0]
    first_solution, owned = load_snapshot(first_snapshot)

    summary = imas.IDSFactory().summary()

    try:
        extracted = build_summary_export_metadata(first_solution, metadata, times=times)
        populate_summary_ids(
            summary,
            metadata,
            extracted,
            time_values=times,
            workflow="time_dependent_discharge",
            description_prefix="Exported full discharge from SOLEDGE-HDG by hdg_postprocess.",
        )
    finally:
        release_snapshot(first_solution, owned)

    return summary


def build_discharge_plasma_profiles_ids(timed_snapshots, metadata: IMASExportMetadata):
    import imas

    first_time, first_snapshot, first_grid = timed_snapshots[0]
    first_solution, owned = load_snapshot(first_snapshot)
    times = np.asarray([time_value for time_value, _, _ in timed_snapshots], dtype=float)

    plasma = imas.IDSFactory().plasma_profiles()
    plasma.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    plasma.time = times

    plasma.grid_ggd.resize(len(timed_snapshots))
    for grid_index, (time_value, _, grid) in enumerate(timed_snapshots, start=1):
        populate_rectangular_grid_ggd_entry(
            plasma.grid_ggd[grid_index - 1],
            grid,
            float(time_value),
            grid_name="rectangular_rz",
            grid_index=grid_index,
        )

    plasma.ggd.resize(len(timed_snapshots))
    for index, (time_value, snapshot, grid) in enumerate(timed_snapshots):
        solution, snapshot_owned = load_snapshot(snapshot)
        try:
            sampled = sample_plasma_fields(solution, grid)
            plasma.ggd[index].time = float(time_value)
            populate_plasma_ggd(plasma.ggd[index], sampled, grid_index=index + 1)
        finally:
            release_snapshot(solution, snapshot_owned)

    try:
        set_constant_zeff(plasma, first_solution, count=len(timed_snapshots))

        plasma.code.name = "SOLEDGE-HDG"
        plasma.code.repository = "hdg_postprocess"
        plasma.code.description = "Time-resolved plasma profiles exported from SOLEDGE-HDG by hdg_postprocess."
        plasma.code.parameters = json.dumps(
            build_discharge_plasma_profiles_metadata(first_solution, first_grid, times),
            sort_keys=True,
        )
    finally:
        release_snapshot(first_solution, owned)

    return plasma


def build_discharge_equilibrium_ids(timed_snapshots, metadata: IMASExportMetadata, *, grid_reference_paths=None):
    import imas

    first_time, first_snapshot, first_grid = timed_snapshots[0]
    first_solution, owned = load_snapshot(first_snapshot)
    times = np.asarray([time_value for time_value, _, _ in timed_snapshots], dtype=float)

    eq = imas.IDSFactory().equilibrium()
    eq.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    eq.time = times

    eq.grids_ggd.resize(len(timed_snapshots))
    for grid_index, (time_value, _, grid) in enumerate(timed_snapshots, start=1):
        populate_discharge_equilibrium_grid(
            eq.grids_ggd[grid_index - 1],
            grid,
            time_value=time_value,
            grid_index=grid_index,
            grid_reference_path=None if grid_reference_paths is None else grid_reference_paths[grid_index - 1],
        )

    eq.time_slice.resize(len(timed_snapshots))
    for index, (time_value, snapshot, grid) in enumerate(timed_snapshots):
        solution, snapshot_owned = load_snapshot(snapshot)
        try:
            sampled_fields = sample_equilibrium_fields(solution, grid)
            eq.time_slice[index].time = float(time_value)
            populate_equilibrium_timeslice(eq.time_slice[index], sampled_fields, grid_index=index + 1)
        finally:
            release_snapshot(solution, snapshot_owned)

    try:
        eq.code.name = "SOLEDGE-HDG"
        eq.code.repository = "hdg_postprocess"
        eq.code.description = "Time-resolved equilibrium exported from SOLEDGE-HDG by hdg_postprocess."

        eq_metadata = build_equilibrium_metadata(first_solution, metadata, first_grid)
        eq_metadata["snapshot_count"] = len(timed_snapshots)
        eq_metadata["exported_times_s"] = times.tolist()
        eq_metadata["grid_count"] = len(timed_snapshots)
        eq_metadata["time_varying_grid"] = True
        eq_metadata["representation_note"] = (
            "Exported as a time-resolved discharge on rectangular cylindrical (R,Z) GGD meshes "
            "through equilibrium.grids_ggd[i]/time_slice[i].ggd."
        )
        if grid_reference_paths is not None:
            eq_metadata["ggd_grid_reference_note"] = (
                "equilibrium.grids_ggd[i] references the topology stored in plasma_profiles.grid_ggd[i] "
                "to avoid duplicate rectangular GGD geometry."
            )
        eq.code.parameters = json.dumps(eq_metadata, sort_keys=True)
    finally:
        release_snapshot(first_solution, owned)

    return eq


def build_discharge_grids(snapshot_count, grid):
    if isinstance(grid, RectangularGrid2D):
        return [grid] * snapshot_count

    grids = list(grid)
    if len(grids) != snapshot_count:
        raise ValueError(
            "When exporting a full discharge with time-varying grids, the grid sequence "
            "must have the same length as the solution sequence."
        )
    for index, one_grid in enumerate(grids):
        if not isinstance(one_grid, RectangularGrid2D):
            raise TypeError(
                f"Grid {index} is not a RectangularGrid2D instance. "
                "The full-discharge exporter currently supports rectangular (R,Z) GGD grids."
            )
    return grids


def build_timed_snapshots(snapshots, grids, *, sort_by_time, time_getter):
    timed_snapshots = []
    previous_time = None

    for index, (snapshot, grid) in enumerate(zip(snapshots, grids)):
        solution, owned = load_snapshot(snapshot)
        try:
            time_value = time_getter(solution)
            if time_value is None:
                raise ValueError(
                    "Full-discharge export requires physically meaningful times for every snapshot. "
                    f"Snapshot {index} does not provide one."
                )
            time_value = float(time_value)
            if not sort_by_time and previous_time is not None and time_value < previous_time:
                raise ValueError(
                    "Full-discharge export expects snapshots to already be ordered in time. "
                    f"Snapshot {index} has time {time_value} s after {previous_time} s."
                )
            timed_snapshots.append((time_value, snapshot, grid))
            previous_time = time_value
        finally:
            release_snapshot(solution, owned)

    if sort_by_time:
        timed_snapshots.sort(key=lambda item: item[0])
    return timed_snapshots


def populate_discharge_equilibrium_grid(
    grids_ggd_entry,
    grid,
    *,
    time_value,
    grid_index,
    grid_reference_path,
):
    if grid_reference_path is None:
        populate_rectangular_grid_ggd_entry(
            grids_ggd_entry,
            grid,
            float(time_value),
            grid_name="rectangular_rz",
            grid_index=grid_index,
        )
        return

    populate_grid_reference_ggd_entry(
        grids_ggd_entry,
        time_value=float(time_value),
        path=str(grid_reference_path),
        grid_name="rectangular_rz",
        grid_index=grid_index,
    )


def build_plasma_grid_reference_paths(occurrence, grid_count):
    return [
        f"#plasma_profiles:{int(occurrence)}/grid_ggd({index})"
        for index in range(1, grid_count + 1)
    ]


def build_written_ids_info(metadata, times):
    return {"occurrence": int(metadata.occurrence), "time_count": len(times)}


def load_snapshot(snapshot):
    if isinstance(snapshot, SolutionSnapshotSource):
        return snapshot.load(), True
    return snapshot, False


def release_snapshot(snapshot, owned):
    if owned:
        del snapshot
        gc.collect()


def release_large_object(value):
    del value
    gc.collect()


def build_discharge_plasma_profiles_metadata(solution, grid, times):
    metadata_dict = build_plasma_profiles_metadata(solution, grid)
    extend_time_metadata(metadata_dict, times)
    return metadata_dict
