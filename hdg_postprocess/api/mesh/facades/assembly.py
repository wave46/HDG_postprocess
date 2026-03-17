import numpy as np

from hdg_postprocess.mesh_operations import boundary as boundary_ops
from hdg_postprocess.mesh_operations import geometry as geometry_ops


class MeshAssembly:
    def __init__(self, mesh):
        self._mesh = mesh

    def full(self):
        if not self._mesh.metadata.flags.combined_to_full:
            geometry_ops.recombine_full_mesh(self._mesh)
        return self._mesh.global_state

    def boundary(self, raw_boundary_info):
        if not self._mesh.metadata.flags.boundary_combined:
            boundary_ops.recombine_full_boundary(self._mesh, raw_boundary_info)
        return self._mesh.boundary_state

    def boundary_gauss(self, boundaries=None, raw_boundary_info=None):
        if boundaries is None:
            if raw_boundary_info is None:
                raise ValueError("Please, provide raw boundary info as input to this method")
            boundaries = np.unique(raw_boundary_info[0]["boundary_flags"])
        boundary_ops.calculate_gauss_boundary(self._mesh, boundaries, raw_boundary_info)
        return self._mesh.boundary_state
