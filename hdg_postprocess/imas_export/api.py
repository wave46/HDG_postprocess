from pathlib import Path

from .config import IMASExportMetadata
from .equilibrium import put_equilibrium
from .plasma_profiles import put_plasma_profiles
from .summary import put_summary


def solution_time_seconds(solution):
    """Return the dimensional solution time in seconds when available."""

    time_group = solution.parameters.get("time", {})
    adim = solution.parameters.get("adimensionalization", {})
    if "Current_time" not in time_group or "time_scale" not in adim:
        return None
    return float(time_group["Current_time"]) * float(adim["time_scale"])


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
