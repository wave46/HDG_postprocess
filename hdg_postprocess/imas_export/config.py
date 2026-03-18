from dataclasses import dataclass


@dataclass
class IMASExportMetadata:
    """User-supplied identifiers and descriptive metadata for one IMAS export."""

    description: str
    shot: int
    run: int
    time: float
    effective_energy_transfer: float
    occurrence: int = 0
    comment: str = ""
    machine: str = ""
