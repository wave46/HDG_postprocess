from hdg_postprocess.mesh_operations import geometry as geometry_ops


class MeshGeometry:
    def __init__(self, mesh):
        self._mesh = mesh

    def recombine_full(self):
        if not self._mesh.metadata.flags.combined_to_full:
            geometry_ops.recombine_full_mesh(self._mesh)
        return self._mesh.global_state

    def ensure_connectivity_big(self):
        if not self._mesh.metadata.flags.connectivity_big_initialized:
            geometry_ops.create_connectivity_big(self._mesh)
        return self._mesh.derived_geometry.connectivity_big

    def ensure_mask(self):
        if not self._mesh.metadata.flags.mask_initialized:
            geometry_ops.make_mask(self._mesh)
        return self._mesh.derived_geometry.mask

    def ensure_element_locator(self):
        if not self._mesh.metadata.flags.element_locator_initialized:
            geometry_ops.make_element_number_function(self._mesh)
        return self._mesh.derived_geometry.element_locator

    def ensure_gauss_volumes(self):
        if not self._mesh.metadata.flags.gauss_volumes_initialized:
            geometry_ops.calculate_gauss_volumes(self._mesh)
        return self._mesh.derived_geometry.gauss_volumes

    def adjacent_elements(self, element_number):
        return geometry_ops.find_adjacent_elements(self._mesh, element_number)
