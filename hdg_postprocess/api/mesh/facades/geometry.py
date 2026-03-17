from hdg_postprocess.mesh_operations import geometry as geometry_ops


class MeshGeometry:
    def __init__(self, mesh):
        self._mesh = mesh

    def _ensure_connectivity_big(self):
        if not self._mesh.metadata.flags.connectivity_big_initialized:
            geometry_ops.create_connectivity_big(self._mesh)

    def _ensure_mask(self):
        if not self._mesh.metadata.flags.mask_initialized:
            geometry_ops.make_mask(self._mesh)

    def _ensure_element_locator(self):
        if not self._mesh.metadata.flags.element_locator_initialized:
            geometry_ops.make_element_number_function(self._mesh)

    def _ensure_gauss_volumes(self):
        if not self._mesh.metadata.flags.gauss_volumes_initialized:
            geometry_ops.calculate_gauss_volumes(self._mesh)

    @property
    def connectivity_big(self):
        self._ensure_connectivity_big()
        return self._mesh.derived_geometry.connectivity_big

    @property
    def mask(self):
        self._ensure_mask()
        return self._mesh.derived_geometry.mask

    @property
    def element_locator(self):
        self._ensure_element_locator()
        return self._mesh.derived_geometry.element_locator

    @property
    def gauss_volumes(self):
        self._ensure_gauss_volumes()
        return self._mesh.derived_geometry.gauss_volumes

    def adjacent_elements(self, element_number):
        return geometry_ops.find_adjacent_elements(self._mesh, element_number)
