import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from raysect.core.math.function.float import Discrete2DMesh
from pathlib import Path
from matplotlib.colors import LogNorm
import os
from hdg_postprocess.mesh_operations import (
    boundary_ordering as boundary_ordering_impl,
    calculate_gauss_boundary as calculate_gauss_boundary_impl,
    calculate_gauss_volumes as calculate_gauss_volumes_impl,
    create_connectivity_big as create_connectivity_big_impl,
    find_adjacent_elements as find_adjacent_elements_impl,
    make_element_number_function as make_element_number_function_impl,
    make_mask as make_mask_impl,
    plot_full_mesh as plot_full_mesh_impl,
    plot_mesh_normals_tangentials as plot_mesh_normals_tangentials_impl,
    plot_mesh_outline as plot_mesh_outline_impl,
    plot_raw_meshes as plot_raw_meshes_impl,
    recombine_full_boundary as recombine_full_boundary_impl,
    recombine_full_mesh as recombine_full_mesh_impl,
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
        
        if self.mesh_parameters['element_type'] == 'triangle':
            if self.mesh_parameters['nodes_per_element']==15:
                self._p_order = 4
            elif self.mesh_parameters['nodes_per_element']==28:
                self._p_order = 6
            elif self.mesh_parameters['nodes_per_element']==45:
                self._p_order = 8
        elif self.mesh_parameters['element_type'] == 'quadrilateral':
            if self.mesh_parameters['nodes_per_element']==49:
                self._p_order = 6
            elif self.mesh_parameters['nodes_per_element']==81:
                self._p_order = 8

        minr,maxr,minz,maxz = 1e5,-1e5,1e5,-1e5
        for vertices in self.raw_vertices:
                       
            minr = min(minr,vertices[:,0].min())
            minz = min(minz,vertices[:,1].min())
            maxr = max(maxr,vertices[:,0].max())
            maxz = max(maxz,vertices[:,1].max())
        self._mesh_extent = {"minr": minr, "maxr":maxr, 
                             "minz": minz, "maxz":maxz}
        self._connectivity_big = None
        self._element_number = None
        self._reference_element = None
        self._vertices_gauss = None
        self._vertices_boundary_gauss = None
        self._volumes_gauss = None
        self._tangentials_gauss = None
        self._normal_guass = None
        self._segment_length_gauss = None
        self._segment_surface_gauss = None
        # not sure if this will be used for serial version
        self._filled = None
        self._indices = None
        self._face_element_number = None
        self._face_local_number = None
        self._face_ghost = None
        if self._n_partitions == 1:
            #no need to combine meshes
            self._combined_to_full = True
            self._connectivity_glob = self.raw_connectivity[0]
            self._connectivity_b_glob = None
            self._vertices_glob = self.raw_vertices[0]
            self._nelems_glob = self._connectivity_glob.shape[0]
            self._nvertices_glob = self._vertices_glob.shape[0]
            self._nfaces_glob = None
            self._boundary_combined = False
            

        
        else:    
            self._combined_to_full = False
            self._connectivity_glob = None
            self._connectivity_b_glob = None
            self._boundary_flags = None
            self._vertices_glob = None
            self._nelems_glob = None
            self._nvertices_glob = None
            self._nfaces_glob = None
            self._boundary_combined = False

            
            
     
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
        return self._vertices_glob

    @property
    def vertices_gauss(self):
        """coordinates of gauss points in global mesh"""
        return self._vertices_gauss
    
    @property
    def vertices_boundary_gauss(self):
        """coordinates of gauss points in global mesh"""
        return self._vertices_boundary_gauss
    
    @property
    def tangentials_gauss(self):
        """tangentials to mesh boundary at gauss points in global mesh"""
        return self._tangentials_gauss
    @property
    def normals_gauss(self):
        """normals to mesh boundary at gauss points in global mesh"""
        return self._normals_gauss

    @property
    def segment_length_gauss(self):
        """segment length corresponding to each gauss point on the boundary"""
        return self._segment_length_gauss
    @property
    def segment_surface_gauss(self):
        """segment length corresponding to each gauss point on the boundary"""
        return self._segment_surface_gauss

    @property
    def volumes_gauss(self):
        """volumes in gauss points in global mesh"""
        return self._volumes_gauss

    @property
    def connectivity_glob(self):
        """connectivity of a global mesh"""
        return self._connectivity_glob

    @property
    def connectivity_b_glob(self):
        """connectivity of faces on global mesh. filled only with boundary faces"""
        return self._connectivity_b_glob
    
    @property
    def boundary_flags(self):
        """
            faces flags on a full mesh.
            -1 means that it is inside boundary, not filled int
            0 boundary between two partitions
            to be done: fill other flags
        """
        return self._boundary_flags

    @property
    def face_element_number(self):
        """
        for each face gives a number of the corresponding element
        """
        return self._face_element_number

    @property
    def face_local_number(self):
        """
        for each face gives a local number of the face in corresponding element
        """
        return self._face_local_number

    @property
    def face_ghost(self):
        """
        If face corresponds to ghost element
        """
        return self._face_ghost

    @property
    def connectivity_big(self):
        """big connectivity of a global mesh for plots
        (each element is triangulated)
        """
        return self._connectivity_big

    @property
    def nelems_glob(self):
        """number of elements in global mesh"""
        return self._nelems_glob

    @property
    def nvertices_glob(self):
        """nubmer of vertices in global mesh"""
        return self._nvertices_glob

    @property
    def nfaces_glob(self):
        """number of faces in global mesh"""
        return self._nfaces_glob

    @property
    def combined_to_full(self):
        """Flag which tells if the mesh has been combined to full"""
        return self._combined_to_full

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
        return self._p_order

    @property
    def mesh_extent(self):
        """Extent of the mesh. A dictionary with minr, maxr, minz and maxz keys."""
        return self._mesh_extent
    
    @property
    def mask(self):
        """Mesh mask which gives 1 if"""
        return self._mask

    @property
    def element_number(self):
        """
        For given pair (R,Z) gives a number of element to which this point relates
        Outside of the mesh gives -1
        """
        return self._element_number
    @element_number.setter
    def element_number(self,value):
        self._element_number = value

    @property
    def boundary_combined(self):
        """Flag which tells if the mesh has been combined on the boundary"""
        return self._boundary_combined

    @property
    def reference_element(self):
        """Dictionary with atomic parameters"""
        return self._reference_element
    @reference_element.setter
    def reference_element(self,value):
        self._reference_element = value

    def plot_raw_meshes(self, data=None, ax=None):
        """
        Plot all raw meshes to a matplotlib figure.
        :param data: Data array defined on the soledgehdg mesh
        """
        return plot_raw_meshes_impl(self, data=data, ax=ax)

    def plot_full_mesh(self, data=None, ax=None, log=False, label=None, connectivity=None, 
                        n_levels=100,limits = None,ticks=None,tick_labels=None,cmap='jet', linewidth=1.0):
        """
        Plot all raw meshes to a matplotlib figure.
        :param data: Data array defined on the soledgehdg mesh
        :param ax: ax where to plot array defined on the soledgehdg mesh
        :param log: if plot in log scale
        :param label: label for the variable
        :param connectivity: if None use the global one, (maybe refined, i.e. each triangle is also triangulated)
        """
        return plot_full_mesh_impl(
            self, data=data, ax=ax, log=log, label=label, connectivity=connectivity,
            n_levels=n_levels, limits=limits, ticks=ticks, tick_labels=tick_labels, cmap=cmap, linewidth=linewidth,
        )

    def plot_mesh_outline(self,raw_boundary_info=None, ax=None):
        """
        Plot mesh outline
        :param raw_boundary_info: is the list of dictio naries with additional boundary info which is saved in solution
        :param ax: ax where to plot 
        """
        return plot_mesh_outline_impl(self, raw_boundary_info=raw_boundary_info, ax=ax)

    def plot_mesh_normals_tangentials(self,raw_boundary_info=None, ax=None,scale=None,scale_units=None):
        """
        Plots quiver plot of tangent and normal vectors to the mesh boundary at the gauss points
        :param raw_boundary_info: is the list of dictio naries with additional boundary info which is saved in solution
        :param ax: ax where to plot
        :param scale: same meaning as in plt.quiver
        :param scale_units: same meaning as in plt.quiver
        by default the size of tangential and normal vectors is normalized to 1 mm on the plot
        It may be adjusted using scale and scale_units settings
        """
        return plot_mesh_normals_tangentials_impl(self, raw_boundary_info=raw_boundary_info, ax=ax, scale=scale, scale_units=scale_units)

    def recombine_full_mesh(self):
        
        recombine_full_mesh_impl(self)
    
    def recombine_full_boundary(self, raw_boundary_info):
        """
        Recombines full boundary connectivity with all needed information to calculate fluxes at the wall.
        Handles cases where boundary types can have disconnected segments or additional closed loops.
        """
        recombine_full_boundary_impl(self, raw_boundary_info)
        
    def boundary_ordering(self, raw_boundary_info, boundaries):
        """
        Recombines ordered boundary connectivity for given boundaries with all needed information to calculate fluxes at the wall.
        Ensures that looped segments are moved to the end of the array.
        """
    
        return boundary_ordering_impl(self, raw_boundary_info, boundaries)




    def create_connectivity_big(self):
        create_connectivity_big_impl(self)

    def make_mask(self):
        """
        to do create interpolator using
        """
        make_mask_impl(self)

    def make_element_number_funtion(self):
        """
        creates a function which gives number of element for givern point
        if outside of the mesh, it gives -1
        """
        make_element_number_function_impl(self)

    def calculate_gauss_volumes(self):
        """
        calculates volumes for each gauss point in the full mesh
        this is neede for volume integration later
        """
        calculate_gauss_volumes_impl(self)

    def calculate_gauss_boundary(self,boundaries,raw_boundary_info):
        """
        calculates:
        vertices in gauss points for boundary faces
        tangential and normal vectors in each point
        segment lengths corresponding to the points
        """
        return calculate_gauss_boundary_impl(self, boundaries, raw_boundary_info)
    def find_adjacent_elements(self,element_number):
        """ 
        finds numbers of adjacent elements
        """
        return find_adjacent_elements_impl(self, element_number)
