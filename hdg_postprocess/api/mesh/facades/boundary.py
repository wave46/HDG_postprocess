from hdg_postprocess.core.mesh import boundary as boundary_ops


class MeshBoundary:
    def __init__(self, mesh):
        self._mesh = mesh

    def ordering(self, raw_boundary_info, boundaries):
        return boundary_ops.boundary_ordering(self._mesh, raw_boundary_info, boundaries)

    def nearest_face_index(self, r, z, raw_boundary_info=None, boundaries=None):
        return boundary_ops.nearest_boundary_face_index(self._mesh, r, z, raw_boundary_info, boundaries)
