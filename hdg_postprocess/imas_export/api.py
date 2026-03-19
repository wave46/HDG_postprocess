from pathlib import Path

from .config import IMASExportMetadata
from .discharge import write_discharge
from .equilibrium import _plasma_grid_reference_path, build_equilibrium_ids, put_equilibrium
from .plasma_profiles import put_plasma_profiles
from .summary import put_summary


def solution_time_seconds(solution, *, allow_steady=False):
    """Return the dimensional solution time in seconds when it is meaningful for export."""

    switches = solution.parameters.get("switches", {})
    if not allow_steady and "steady" in switches and bool(_as_python_scalar(switches["steady"])):
        return None

    time_group = solution.parameters.get("time", {})
    adim = solution.parameters.get("adimensionalization", {})
    if "Current_time" not in time_group or "time_scale" not in adim:
        return None
    return float(_as_python_scalar(time_group["Current_time"])) * float(_as_python_scalar(adim["time_scale"]))


def write_summary_netcdf(solution, path, metadata: IMASExportMetadata, *, file_mode="x"):
    """Write one summary IDS into a netCDF-backed IMAS DBEntry."""

    return _write_netcdf(path, file_mode, lambda entry: put_summary(entry, solution, metadata))


def write_equilibrium_netcdf(solution, path, metadata: IMASExportMetadata, grid, *, file_mode="x"):
    """Write one equilibrium IDS into a netCDF-backed IMAS DBEntry."""

    return _write_netcdf(path, file_mode, lambda entry: put_equilibrium(entry, solution, metadata, grid))


def write_plasma_profiles_netcdf(solution, path, metadata: IMASExportMetadata, grid, *, file_mode="x"):
    """Write one plasma_profiles IDS into a netCDF-backed IMAS DBEntry."""

    return _write_netcdf(path, file_mode, lambda entry: put_plasma_profiles(entry, solution, metadata, grid))


def write_imas_netcdf(
    solution,
    path,
    metadata: IMASExportMetadata,
    grid,
    *,
    file_mode="x",
    include_summary=True,
    include_equilibrium=True,
    include_plasma_profiles=True,
):
    """Write the selected IMAS IDSs into one netCDF-backed DBEntry."""

    return _write_netcdf(
        path,
        file_mode,
        lambda entry: _write_selected_ids(
            entry,
            solution,
            metadata,
            grid,
            include_summary=include_summary,
            include_equilibrium=include_equilibrium,
            include_plasma_profiles=include_plasma_profiles,
        ),
    )


def write_imas_scan_case_netcdf(
    solution,
    path,
    metadata: IMASExportMetadata,
    grid,
    *,
    create=False,
    include_summary=True,
    include_equilibrium=True,
    include_plasma_profiles=True,
):
    """Write one scan point into a bundled netCDF-backed DBEntry using its occurrence index."""

    return write_imas_netcdf(
        solution,
        path,
        metadata,
        grid,
        file_mode="w" if create else "a",
        include_summary=include_summary,
        include_equilibrium=include_equilibrium,
        include_plasma_profiles=include_plasma_profiles,
    )


def write_discharge_imas_netcdf(
    solutions,
    path,
    metadata: IMASExportMetadata,
    grid,
    *,
    file_mode="x",
    include_summary=True,
    include_equilibrium=True,
    include_plasma_profiles=True,
    sort_by_time=False,
    time_getter=None,
):
    """Write a full time-resolved discharge into one netCDF-backed DBEntry.

    The default path expects snapshots to already be ordered in time and will
    raise if they are not. Pass ``sort_by_time=True`` only when you explicitly
    want the exporter to reorder snapshots.
    """

    if time_getter is None:
        time_getter = solution_time_seconds

    return _write_netcdf(
        path,
        file_mode,
        lambda entry: write_discharge(
            entry,
            solutions,
            metadata,
            grid,
            include_summary=include_summary,
            include_equilibrium=include_equilibrium,
            include_plasma_profiles=include_plasma_profiles,
            sort_by_time=sort_by_time,
            time_getter=time_getter,
        ),
    )


def _prepare_db_path(path):
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _write_netcdf(path, file_mode, writer):
    import imas

    with imas.DBEntry(str(_prepare_db_path(path)), file_mode) as entry:
        return writer(entry)


def _write_selected_ids(
    entry,
    solution,
    metadata,
    grid,
    *,
    include_summary,
    include_equilibrium,
    include_plasma_profiles,
):
    written = {}

    if include_summary:
        written["summary"] = put_summary(entry, solution, metadata)

    if include_plasma_profiles:
        written["plasma_profiles"] = put_plasma_profiles(entry, solution, metadata, grid)

    if include_equilibrium:
        written["equilibrium"] = _put_equilibrium_for_selected_ids(
            entry,
            solution,
            metadata,
            grid,
            include_plasma_profiles=include_plasma_profiles,
        )

    return written


def _put_equilibrium_for_selected_ids(entry, solution, metadata, grid, *, include_plasma_profiles):
    if not include_plasma_profiles:
        return put_equilibrium(entry, solution, metadata, grid)

    equilibrium = build_equilibrium_ids(
        solution,
        metadata,
        grid,
        grid_reference_path=_plasma_grid_reference_path(
            occurrence=metadata.occurrence,
            grid_index=1,
        ),
    )
    entry.put(equilibrium, metadata.occurrence)
    return equilibrium


def _as_python_scalar(value):
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            if value.size == 1:
                return value.reshape(-1)[0].item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    return value
