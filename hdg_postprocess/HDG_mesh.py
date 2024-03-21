import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from raysect.core.math.function.float import Discrete2DMesh
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
        self._connectivity_b_glob = None
        self._boundary_flags = None
        self._face_element_number = None
        self._face_local_number = None
        self._face_ghost = None
        self._vertices_glob = None
        self._nelems_glob = None
        self._nvertices_glob = None
        self._nfaces_glob = None
        self._connectivity_big = None
        self._element_number = None
        self._reference_element = None
        self._element_number_mask = None
        self._vertices_gauss = None
        self._volumes_gauss = None
        self._filled = None
        self._indices = None
     
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

    @property
    def reference_element(self):
        """Dictionary with atomic parameters"""
        return self._reference_element
    @reference_element.setter
    def reference_element(self,value):
        self._reference_element = value

    @property
    def element_number_mask(self):
        """Mask which gives a number of element for a given (R,Z)"""
        return self._element_number_mask
    @element_number_mask.setter
    def element_number_mask(self,value):
        self._element_number_mask = value

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
    
    def recombine_full_boundary(self,raw_boundary_info):
        """
        Recombines full boundary connectivity with all needed information to calculate fluxes at the wall
        """
        self._nfaces_glob = 0
        for i in range(self.n_partitions):
            self._nfaces_glob = max(self._nfaces_glob,self.raw_rest_mesh_data[i]['loc2glob_fa'].max())
        
        connectivity_b_glob = -1*np.ones((self._nfaces_glob+1,self.mesh_parameters['nodes_per_face']),dtype=int)
        boundary_flags = -1*np.ones((self._nfaces_glob+1),dtype=int)
        face_element_number = np.zeros((self._nfaces_glob+1,1),dtype=int)
        face_local_number = np.zeros((self._nfaces_glob+1),dtype=int)

        for i in range(self.n_partitions):
            non_ghost = (~self.raw_ghost_faces[i].flatten())[-self.raw_mesh_numbers[i]['Nextfaces']:]

            connectivity_b_glob[self.raw_rest_mesh_data[i]['loc2glob_fa'][-self.raw_mesh_numbers[i]['Nextfaces']:][non_ghost],:] = \
                self.raw_rest_mesh_data[i]['loc2glob_no'][self.raw_connectivity_boundary[i][:,:]][non_ghost]
            boundary_flags[self.raw_rest_mesh_data[i]['loc2glob_fa'][-self.raw_mesh_numbers[i]['Nextfaces']:][non_ghost]] = \
                raw_boundary_info[i]['boundary_flags'][non_ghost]
            face_element_number[self.raw_rest_mesh_data[i]['loc2glob_fa'][-self.raw_mesh_numbers[i]['Nextfaces']:][non_ghost]] = \
                self.raw_rest_mesh_data[i]['loc2glob_el'][raw_boundary_info[i]['exterior_faces'][:,0][non_ghost]][None].T
            face_local_number[self.raw_rest_mesh_data[i]['loc2glob_fa'][-self.raw_mesh_numbers[i]['Nextfaces']:][non_ghost]] = \
                raw_boundary_info[i]['exterior_faces'][:,1][non_ghost]

        # we filled in only exterior faces info
        # excluding empty entrances
        filled = (connectivity_b_glob!=-1).all(axis=1)
        connectivity_b_glob = connectivity_b_glob[filled,:]
        boundary_flags = boundary_flags[filled]
        face_element_number = face_element_number[filled]
        face_local_number = face_local_number[filled]

        #save to make easier skeleton solution rebuilding
        self._filled = filled

        # this boundary is not ordered, so now we reorder it
        indices = [0]
        i = 0
        while(len(indices)!=len(connectivity_b_glob)):
            i = np.where(connectivity_b_glob[i,-1]==connectivity_b_glob[:,0])[0][0]
            indices.append(i)
        self._connectivity_b_glob = connectivity_b_glob[indices,:]
        self._boundary_flags = boundary_flags[indices]
        self._face_element_number = face_element_number[indices]
        self._face_local_number = face_local_number[indices]

        #save to make easier skeleton solution rebuilding
        self._indices = indices


        




    def create_connectivity_big(self):
        #additional triangulaton: since we have more than 3 points in each element, we can triangulate it
        #take any triangle from the mesh
        if not self._combined_to_full:
            self.recombine_full_mesh()
        # defining triangulation order of each big triangle
        if self.mesh_parameters["element_type"] == "triangle":
            if self.p_order == 4:
                triangle_indexes = np.array([[1,4,12],
                        [4,5,13],
                        [5,6,14],
                        [6,2,7],
                        [7,8,14],
                        [8,9,15],
                        [9,3,10],
                        [10,11,15],
                        [11,12,13],
                        [12,4,13],
                        [14,6,7],
                        [13,5,14],
                        [13,14,8],
                        [13,8,15],
                        [15,9,10],
                        [11,13,15]])
                triangle_indexes -=1
            else:
                raise KeyError(f'{self.p_order} splitting of each element is not defined yet')
        else:
            raise KeyError(f'{self.mesh_parameters["element_type"]} splitting of each element is not defined yet')
                
        connectivity_big = self.connectivity_glob[:,triangle_indexes]
        self._connectivity_big = connectivity_big.reshape(connectivity_big.shape[0]*connectivity_big.shape[1],3)

    def make_mask(self):
        """
        to do create interpolator using
        """
        if self.connectivity_big is None:
            self.create_connectivity_big()        
        
        self._mask = Discrete2DMesh(self.vertices_glob, self.connectivity_big,
                     np.ones(self.connectivity_big.shape[0]), limit=False, default_value=0)

    def make_element_number_funtion(self):
        """
        creates a function which gives number of element for givern point
        if outside of the mesh, it gives -1
        """
        if self.connectivity_big is None:
            self.create_connectivity_big()
        element_numbers = np.repeat(np.arange(len(self.connectivity_glob)),self.connectivity_big.shape[0]/self.connectivity_glob.shape[0])
        self._element_number = Discrete2DMesh(self.vertices_glob, self.connectivity_big,
                                 element_numbers,limit=False,default_value = -1)

    def calculate_gauss_volumes(self):
        """
        calculates volumes for each gauss point in the full mesh
        this is neede for volume integration later
        """
        if self.reference_element is None:
            raise ValueError("Please, provide reference element")
        if not self._combined_to_full:
            self.recombine_full_mesh()

        self._vertices_gauss = np.einsum('ij,kjh->kih', self.reference_element['N'],self.vertices_glob[self.connectivity_glob,:])
        J11_loc = np.einsum('ij,kj->ki',self.reference_element['Nxi'],self.vertices_glob[self.connectivity_glob,0])
        J12_loc = np.einsum('ij,kj->ki',self.reference_element['Nxi'],self.vertices_glob[self.connectivity_glob,1])
        J21_loc = np.einsum('ij,kj->ki',self.reference_element['Neta'],self.vertices_glob[self.connectivity_glob,0])
        J22_loc = np.einsum('ij,kj->ki',self.reference_element['Neta'],self.vertices_glob[self.connectivity_glob,1])
        detJ_loc = J11_loc*J22_loc-J12_loc*J21_loc

        self._volumes_gauss = 2*np.pi*self.reference_element['IPweights'][None,:]*detJ_loc[:,:]*self._vertices_gauss[:,:,0]
        