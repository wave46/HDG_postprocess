import numpy as np
from hdg_postprocess.mesh_api import (
    MeshBoundary,
    MeshBoundaryState,
    MeshDerivedGeometryState,
    MeshGeometry,
    MeshGlobalState,
    MeshMetadata,
    MeshPlot,
)
class HDGmesh:
    """
    SOLEDGE-HDG mesh object  
    The mesh is triangular (so far), high order  (so far p=4 or p=6), so it can have more than 3 nodes per element
    """

    def __init__(self, raw_vertices,raw_connectivity,raw_connectivity_boundary,
                raw_mesh_numbers,raw_boundary_flags,raw_ghost_elements,
                raw_ghost_faces,mesh_parameters,n_partitions,raw_rest_mesh_data=None):
        """
        Here we just store the raw data
        """
        self._raw_vertices = raw_vertices
        self._raw_connectivity = raw_connectivity
        self._raw_connectivity_boundary = raw_connectivity_boundary
        self._raw_mesh_numbers = raw_mesh_numbers
        self._raw_boundary_flags = raw_boundary_flags
        self._raw_ghost_elements = raw_ghost_elements
        self._raw_ghost_faces = raw_ghost_faces
        self._mesh_parameters = mesh_parameters
        self._n_partitions = n_partitions
        if n_partitions>1:
            if raw_rest_mesh_data is None:
                raise ValueError("communication info is not provided")

            self._raw_rest_mesh_data = raw_rest_mesh_data

        self._initial_setup()



    def _initial_setup(self):
        self._boundary = MeshBoundary(self)
        self._geometry = MeshGeometry(self)
        self._plot = MeshPlot(self)
        self._metadata = MeshMetadata()
        self._global_state = MeshGlobalState()
        self._derived_geometry = MeshDerivedGeometryState()
        self._boundary_state = MeshBoundaryState()

        if self.mesh_parameters['element_type'] == 'triangle':
            if self.mesh_parameters['nodes_per_element']==15:
                self._metadata.p_order = 4
            elif self.mesh_parameters['nodes_per_element']==28:
                self._metadata.p_order = 6
            elif self.mesh_parameters['nodes_per_element']==45:
                self._metadata.p_order = 8
        elif self.mesh_parameters['element_type'] == 'quadrilateral':
            if self.mesh_parameters['nodes_per_element']==49:
                self._metadata.p_order = 6
            elif self.mesh_parameters['nodes_per_element']==81:
                self._metadata.p_order = 8

        minr,maxr,minz,maxz = 1e5,-1e5,1e5,-1e5
        for vertices in self.raw_vertices:
                       
            minr = min(minr,vertices[:,0].min())
            minz = min(minz,vertices[:,1].min())
            maxr = max(maxr,vertices[:,0].max())
            maxz = max(maxz,vertices[:,1].max())
        self._metadata.extent = {"minr": minr, "maxr":maxr, 
                                 "minz": minz, "maxz":maxz}
        if self._n_partitions == 1:
            #no need to combine meshes
            self._metadata.flags.combined_to_full = True
            self._global_state.connectivity = self.raw_connectivity[0]
            self._global_state.vertices = self.raw_vertices[0]
            self._global_state.n_elements = self._global_state.connectivity.shape[0]
            self._global_state.n_vertices = self._global_state.vertices.shape[0]
            self._global_state.n_faces = None
            self._metadata.flags.boundary_combined = False
            

        
        else:    
            self._metadata.flags.combined_to_full = False
            self._global_state.connectivity = None
            self._global_state.vertices = None
            self._global_state.n_elements = None
            self._global_state.n_vertices = None
            self._global_state.n_faces = None
            self._metadata.flags.boundary_combined = False

            
            
     
    @property
    def raw_vertices(self):
        """raw vertices"""
        return self._raw_vertices

    @property
    def raw_connectivity(self):
        """raw connectivity"""
        return self._raw_connectivity

    @property
    def vertices_glob(self):
        """vertices in global mesh"""
        return self._global_state.vertices

    @property
    def vertices_gauss(self):
        """coordinates of gauss points in global mesh"""
        return self._derived_geometry.vertices_gauss

    @property
    def connectivity_glob(self):
        """connectivity of a global mesh"""
        return self._global_state.connectivity

    @property
    def connectivity_b_glob(self):
        """connectivity of faces on global mesh. filled only with boundary faces"""
        return self._boundary_state.connectivity
    
    @property
    def boundary_flags(self):
        """
            faces flags on a full mesh.
            -1 means that it is inside boundary, not filled int
            0 boundary between two partitions
            to be done: fill other flags
        """
        return self._boundary_state.flags

    @property
    def face_element_number(self):
        """
        for each face gives a number of the corresponding element
        """
        return self._boundary_state.face_element_number

    @property
    def face_local_number(self):
        """
        for each face gives a local number of the face in corresponding element
        """
        return self._boundary_state.face_local_number

    @property
    def face_ghost(self):
        """
        If face corresponds to ghost element
        """
        return None

    @property
    def nelems_glob(self):
        """number of elements in global mesh"""
        return self._global_state.n_elements

    @property
    def nvertices_glob(self):
        """nubmer of vertices in global mesh"""
        return self._global_state.n_vertices

    @property
    def nfaces_glob(self):
        """number of faces in global mesh"""
        return self._global_state.n_faces

    @property
    def raw_connectivity_boundary(self):
        """raw connectivity at the boundary"""
        return self._raw_connectivity_boundary
    
    @property
    def raw_mesh_numbers(self):
        """raw mesh numbers dictionary"""
        return self._raw_mesh_numbers

    @property
    def raw_boundary_flags(self):
        """raw mesh boundary flags"""
        return self._raw_boundary_flags

    @property
    def raw_ghost_elements(self):
        """raw mesh ghost elements flags"""
        return self._raw_ghost_elements

    @property
    def raw_ghost_faces(self):
        """raw mesh ghost elements flags"""
        return self._raw_ghost_faces

    @property
    def raw_rest_mesh_data(self):
        """raw rest mesh data"""
        return self._raw_rest_mesh_data

    @property
    def mesh_parameters(self):
        """mesh parameters"""
        return self._mesh_parameters

    @property
    def n_partitions(self):
        """number of partitions"""
        return self._n_partitions

    @property
    def p_order(self):
        """polynomial order of the mesh"""
        return self._metadata.p_order

    @property
    def mesh_extent(self):
        """Extent of the mesh. A dictionary with minr, maxr, minz and maxz keys."""
        return self._metadata.extent

    @property
    def geometry(self):
        """Facade for recombination and derived mesh geometry."""
        return self._geometry

    @property
    def boundary(self):
        """Facade for boundary recombination, ordering, and gauss geometry."""
        return self._boundary

    @property
    def plot(self):
        """Facade for mesh plotting workflows."""
        return self._plot

    @property
    def metadata(self):
        """Structured metadata, flags, and cache state for the mesh."""
        return self._metadata

    @property
    def global_state(self):
        """Structured full-mesh state."""
        return self._global_state

    @property
    def derived_geometry(self):
        """Structured derived geometry state and caches."""
        return self._derived_geometry

    @property
    def boundary_state(self):
        """Structured boundary and boundary-gauss state."""
        return self._boundary_state
