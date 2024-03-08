import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm
class HDGmesh:
    """
    SOLEDGE-HDG mesh object  
    The mesh is triangular (so far), high order  (so far p=4 or p=6), so it can have more than 3 nodes per element
    """

    def __init__(self, raw_vertices,raw_connectivity,raw_connectivity_boundary,
                raw_mesh_numbers,raw_boundary_flags,raw_ghost_elements,
                raw_ghost_faces,raw_rest_mesh_data,mesh_parameters,n_partitions):
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
        self._raw_rest_mesh_data = raw_rest_mesh_data
        self._mesh_parameters = mesh_parameters
        self._n_partitions = n_partitions

        self._initial_setup()



    def _initial_setup(self):
        
        if self.mesh_parameters['element_type'] == 'triangle':
            if self.mesh_parameters['nodes_per_element']==15:
                self._p_order = 4

        minr,maxr,minz,maxz = 1e5,-1e5,1e5,-1e5
        for vertices in self.raw_vertices:
                       
            minr = min(minr,vertices[:,0].min())
            minz = min(minz,vertices[:,1].min())
            maxr = max(maxr,vertices[:,0].max())
            maxz = max(maxz,vertices[:,1].max())
        self._mesh_extent = {"minr": minr, "maxr":maxr, 
                             "minz": minz, "maxz":maxz}

        self._combined_to_full = False
        self._connectivity_glob = None
        self._vertices_glob = None
        self._nelems_glob = None
        self._nvertices_glob = None
     
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
    def connectivity_glob(self):
        """connectivity of a global mesh"""
        return self._connectivity_glob

    @property
    def nelems_glob(self):
        """number of elements in global mesh"""
        return self._nelems_glob

    @property
    def nvertices_glob(self):
        """nubmer of vertices in global mesh"""
        return self._nvertices_glob

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

    def plot_raw_meshes(self, data=None, ax=None):
        """
        Plot all raw meshes to a matplotlib figure.
        :param data: Data array defined on the soledgehdg mesh
        """
        colors = cm.get_cmap('hsv', self.n_partitions)
        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)
        for i,(vertices, connectivity) in enumerate(zip(self.raw_vertices,self.raw_connectivity)):
            if data is None:            
                verts = vertices[connectivity[:,:3]]
                collection = PolyCollection(verts, facecolor="none", edgecolor=colors(i), linewidth=0.05)

            else:
                verts = vertices[connectivity[:,:3]]
                collection= PolyCollection(verts)
                collection.set_array(data[i][connectivity[:,:3]].mean(axis = 1))            
            ax.add_collection(collection)        
        ax.set_aspect(1)
        ax.set_xlim(self.mesh_extent["minr"], self.mesh_extent["maxr"])
        ax.set_ylim(self.mesh_extent["minz"], self.mesh_extent["maxz"])
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        return ax

    def plot_full_mesh(self, data=None, ax=None, log=False, label=None, connectivity=None, n_levels=100):
        """
        Plot all raw meshes to a matplotlib figure.
        :param data: Data array defined on the soledgehdg mesh
        :param ax: ax where to plot array defined on the soledgehdg mesh
        :param log: if plot in log scale
        :param label: label for the variable
        :param connectivity: if None use the global one, (maybe refined, i.e. each triangle is also triangulated)
        """
        if (not self._combined_to_full):
            
            print('Comibining to full mesh')
            self.recombine_full_mesh()

        colors = 'b'
        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        if data is None:            
            verts = self.vertices_glob[self.connectivity_glob[:,:3]]
            collection = PolyCollection(verts, facecolor="none", edgecolor=colors, linewidth=0.05)
            ax.add_collection(collection) 
        else:
            if connectivity is None:
                connectivity = self.connectivity_glob[:,:3]
            if log:
                im = ax.tricontourf(self.vertices_glob[:,0], self.vertices_glob[:,1], np.log10(data),levels=n_levels
                    , extend='both',triangles = connectivity, cmap='jet'
                    ,extendrect = True)
                ax.set_title(f'log10({label})')      
            else:
                im = ax.tricontourf(self.vertices_glob[:,0], self.vertices_glob[:,1], data,levels=n_levels
                    , extend='both',triangles = connectivity, cmap='jet'
                    ,extendrect = True)
                ax.set_title(f'{label}')     
            cbar = plt.colorbar(im, ax=ax,extendrect = True)
               
        ax.set_aspect(1)
        ax.set_xlim(self.mesh_extent["minr"], self.mesh_extent["maxr"])
        ax.set_ylim(self.mesh_extent["minz"], self.mesh_extent["maxz"])
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        return ax

    def recombine_full_mesh(self):
        
        self._nelems_glob = 0
        self._nvertices_glob = 0

        for i in range(self.n_partitions):
            self._nelems_glob = max(self._nelems_glob, self.raw_rest_mesh_data[i]['loc2glob_el'].max())
            self._nvertices_glob = max(self._nvertices_glob, self.raw_rest_mesh_data[i]['loc2glob_no'].max())

        self._connectivity_glob = np.zeros((self._nelems_glob+1,self.mesh_parameters['nodes_per_element']),dtype=int)
        self._vertices_glob = np.zeros((self._nvertices_glob+1,self.mesh_parameters['Ndim']))
        for i in range(self.n_partitions):
            # removing ghost elements as well
            self._connectivity_glob[self.raw_rest_mesh_data[i]['loc2glob_el'][~self.raw_ghost_elements[i].astype(bool).flatten()]] = \
                self.raw_rest_mesh_data[i]['loc2glob_no'][self.raw_connectivity[i][~self.raw_ghost_elements[i].astype(bool).flatten(),:]]
            self._vertices_glob[self.raw_rest_mesh_data[i]['loc2glob_no'],:] = self.raw_vertices[i][:,:]
        self._combined_to_full = True


