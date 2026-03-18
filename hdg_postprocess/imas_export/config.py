from dataclasses import dataclass
import math

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

    @classmethod
    def from_solution_bounds(cls, solution, *, dr=0.003, dz=0.003, padding=0.0):
        if solution.mesh.global_state.vertices is None:
            solution.mesh.assembly.full()
        vertices = solution.mesh.global_state.vertices
        r_min = float(vertices[:, 0].min()) - padding
        r_max = float(vertices[:, 0].max()) + padding
        z_min = float(vertices[:, 1].min()) - padding
        z_max = float(vertices[:, 1].max()) + padding
        nr = max(2, int(math.ceil((r_max - r_min) / dr)) + 1)
        nz = max(2, int(math.ceil((z_max - z_min) / dz)) + 1)
        return cls(r_min=r_min, r_max=r_max, nr=nr, z_min=z_min, z_max=z_max, nz=nz)

    def axes(self):
        r_axis = np.linspace(self.r_min, self.r_max, self.nr)
        z_axis = np.linspace(self.z_min, self.z_max, self.nz)
        return r_axis, z_axis

    def mesh(self):
        r_axis, z_axis = self.axes()
        return np.meshgrid(r_axis, z_axis, indexing="ij")

    def flattened_points(self):
        r_grid, z_grid = self.mesh()
        return r_grid.reshape(-1), z_grid.reshape(-1)
