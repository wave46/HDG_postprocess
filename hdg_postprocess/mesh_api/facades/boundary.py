from hdg_postprocess.mesh_operations import boundary as boundary_ops


class MeshBoundary:
    def __init__(self, mesh):
        self._mesh = mesh

    def recombine_full(self, raw_boundary_info):
        boundary_ops.recombine_full_boundary(self._mesh, raw_boundary_info)
        return self._mesh.connectivity_b_glob

    def ordering(self, raw_boundary_info, boundaries):
        return boundary_ops.boundary_ordering(self._mesh, raw_boundary_info, boundaries)

    def gauss(self, boundaries, raw_boundary_info):
        return boundary_ops.calculate_gauss_boundary(self._mesh, boundaries, raw_boundary_info)
