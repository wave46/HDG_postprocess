from dataclasses import dataclass

import numpy as np


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
    poloidal_flux_convention: str = ""


@dataclass
class RectangularGrid2D:
    """Rectangular (R, Z) sampling grid used by the first IMAS exporters."""

    r_min: float
    r_max: float
    nr: int
    z_min: float
    z_max: float
    nz: int

    def axes(self):
        r_axis = np.linspace(self.r_min, self.r_max, self.nr)
        z_axis = np.linspace(self.z_min, self.z_max, self.nz)
        return r_axis, z_axis

    def mesh(self):
        r_axis, z_axis = self.axes()
        return np.meshgrid(r_axis, z_axis, indexing="ij")
