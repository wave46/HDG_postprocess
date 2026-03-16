from hdg_postprocess.mesh_operations import geometry as geometry_ops


class MeshGeometry:
    def __init__(self, mesh):
        self._mesh = mesh

    def recombine_full(self):
        geometry_ops.recombine_full_mesh(self._mesh)
        return self._mesh.vertices_glob, self._mesh.connectivity_glob

    def connectivity_big(self):
        geometry_ops.create_connectivity_big(self._mesh)
        return self._mesh.connectivity_big

    def mask(self):
        geometry_ops.make_mask(self._mesh)
        return self._mesh.mask

    def element_locator(self):
        geometry_ops.make_element_number_function(self._mesh)
        return self._mesh.element_number

    def gauss_volumes(self):
        geometry_ops.calculate_gauss_volumes(self._mesh)
        return self._mesh.volumes_gauss

    def adjacent_elements(self, element_number):
        return geometry_ops.find_adjacent_elements(self._mesh, element_number)
