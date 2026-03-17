from hdg_postprocess.mesh_operations import boundary as boundary_ops


class MeshBoundary:
    def __init__(self, mesh):
        self._mesh = mesh

    def ordering(self, raw_boundary_info, boundaries):
        return boundary_ops.boundary_ordering(self._mesh, raw_boundary_info, boundaries)
