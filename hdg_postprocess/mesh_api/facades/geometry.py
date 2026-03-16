from hdg_postprocess.mesh_operations import geometry as geometry_ops


class MeshGeometry:
    def __init__(self, mesh):
        self._mesh = mesh

    def recombine_full(self):
        if not self._mesh.combined_to_full:
            geometry_ops.recombine_full_mesh(self._mesh)
        return self._mesh.vertices_glob, self._mesh.connectivity_glob

    @property
    def connectivity_big(self):
        if self._mesh.connectivity_big is None:
            geometry_ops.create_connectivity_big(self._mesh)
        return self._mesh.connectivity_big

    @property
    def mask(self):
        if self._mesh.mask is None:
            geometry_ops.make_mask(self._mesh)
        return self._mesh.mask

    @property
    def element_locator(self):
        if self._mesh.element_number is None:
            geometry_ops.make_element_number_function(self._mesh)
        return self._mesh.element_number

    @property
    def gauss_volumes(self):
        if self._mesh.volumes_gauss is None:
            geometry_ops.calculate_gauss_volumes(self._mesh)
        return self._mesh.volumes_gauss

    def adjacent_elements(self, element_number):
        return geometry_ops.find_adjacent_elements(self._mesh, element_number)
