from pathlib import Path

from .config import IMASExportMetadata
from .equilibrium import put_equilibrium
from .plasma_profiles import put_plasma_profiles
from .summary import put_summary


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

    import imas

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with imas.DBEntry(str(db_path), file_mode) as entry:
        return put_summary(entry, solution, metadata)


def write_equilibrium_netcdf(solution, path, metadata: IMASExportMetadata, grid, *, file_mode="x"):
    """Write one equilibrium IDS into a netCDF-backed IMAS DBEntry."""

    import imas

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with imas.DBEntry(str(db_path), file_mode) as entry:
        return put_equilibrium(entry, solution, metadata, grid)


def write_plasma_profiles_netcdf(solution, path, metadata: IMASExportMetadata, grid, *, file_mode="x"):
    """Write one plasma_profiles IDS into a netCDF-backed IMAS DBEntry."""

    import imas

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with imas.DBEntry(str(db_path), file_mode) as entry:
        return put_plasma_profiles(entry, solution, metadata, grid)


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

    import imas

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    written = {}
    with imas.DBEntry(str(db_path), file_mode) as entry:
        if include_summary:
            written["summary"] = put_summary(entry, solution, metadata)
        if include_equilibrium:
            written["equilibrium"] = put_equilibrium(entry, solution, metadata, grid)
        if include_plasma_profiles:
            written["plasma_profiles"] = put_plasma_profiles(entry, solution, metadata, grid)
    return written


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

    file_mode = "w" if create else "a"
    return write_imas_netcdf(
        solution,
        path,
        metadata,
        grid,
        file_mode=file_mode,
        include_summary=include_summary,
        include_equilibrium=include_equilibrium,
        include_plasma_profiles=include_plasma_profiles,
    )
