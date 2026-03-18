from pathlib import Path

from .config import IMASExportMetadata
from .equilibrium import put_equilibrium
from .summary import put_summary


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
