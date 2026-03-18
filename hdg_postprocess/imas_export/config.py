from dataclasses import dataclass


@dataclass
class IMASExportMetadata:
    """User-supplied identifiers and descriptive metadata for one IMAS export."""

    description: str
    shot: int
    run: int
    time: float
    occurrence: int = 0
    comment: str = ""
    case_name: str = ""
    machine: str = ""
