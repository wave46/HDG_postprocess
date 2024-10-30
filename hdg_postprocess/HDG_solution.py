import numpy as np
import matplotlib.pyplot as plt
from hdg_postprocess.routines.atomic import *
from hdg_postprocess.routines.plasma import *
from hdg_postprocess.routines.neutrals import *
from raysect.core.math.function.float import Discrete2DMesh
from hdg_postprocess.routines.interpolators import SoledgeHDG2DInterpolator
import os
class HDGsolution:
    ""
    
    def __init__(self,raw_solutions, raw_solutions_skeleton, raw_gradients,
                 raw_equilibriums,raw_solution_boundary_infos, parameters, 
                 n_partitions, mesh):
        self._parameters = parameters
        self._neq = parameters['Neq'][0]
        self._nphys = len(self.parameters['physics']['physical_variable_names']) 
        self._ndim = parameters['Ndim'][0]
        self._n_partitions = n_partitions
        self._raw_equilibriums = raw_equilibriums
        self._raw_solution_boundary_infos = raw_solution_boundary_infos
        self._mesh = mesh


        self._raw_solutions = []
        self._raw_solutions_skeleton = []
        self._raw_gradients = []

        for raw_solution,raw_solution_skeleton,raw_gradient \
            in zip(raw_solutions,raw_solutions_skeleton,raw_gradients):

            self._raw_solutions.append(raw_solution.reshape(raw_solution.shape[0]//self.neq,self.neq))
            self._raw_solutions_skeleton.append(raw_solution_skeleton.reshape(raw_solution_skeleton.shape[0]//self.neq,self.neq))
            raw_gradient = raw_gradient.reshape(raw_gradient.shape[0]//(self.neq*self.ndim), (self.neq*self.ndim))
            raw_gradient = raw_gradient.reshape(raw_gradient.shape[0],self.neq,self.ndim)
            self._raw_gradients.append(raw_gradient)

        self._initial_setup()

    def _initial_setup(self):
        # simple representation of a solution
        self._combined_simple_solution = False
        self._solution_simple = None
        self._gradient_simple = None
        self._magnetic_field_simple= None
        #physical solution flags
        self._full_phys_initialized = False
        self._simple_phys_initialized = False

        self._solution_simple_phys = None
        self._gradient_simple_phys = None

        self._solution_glob_phys = None
        self._gradient_glob_phys = None

        #boundary solutions
        self._solution_boundary = None
        self._solution_skeleton_boundary = None
        self._gradient_boundary = None
        
        # neutral parameters
        self._atomic_parameters = None
        self._dnn_parameters = None
        self._ionization_source_simple = None
        self._cx_source_simple = None
        self._ionization_rate_simple = None   
        self._recombination_rate_simple = None        
        self._cx_rate_simple = None
        self._dnn_simple = None
        self._dnn_with_nn_collision_simple = None
        self._mfp_simple = None

        #plasma parameters
        self._ohmic_source_simple = None

        #magentic field parameters
        self._r_axis = None
        self._z_axis = None
        self._a_simple = None
        self._a_glob = None
        self._qcyl_simple = None
        self._qcyl_glob = None

        #turbulent model parameters
        self._dk_parameters = None
        self._dk_simple = None
        self._dk_glob = None


        #interpolators
        self._sample_interpolator = None
        self._solution_interpolators= None
        self._gradient_interpolators= None

        #values in gauss points

        self._solution_gauss = None
        self._gradient_gauss = None
        self._magnetic_field_gauss = None
        self._jtor_gauss = None
        self._ohmic_source_gauss = None
        self._ionization_source_gauss = None
        self._cx_source_gauss = None

        # defining the indexes of conservative variables
        self._cons_idx = {}
        for i,label in enumerate(self.parameters['physics']['conservative_variable_names']):
            self._cons_idx[label] = i
        self._phys_idx = {}
        for i,label in enumerate(self.parameters['physics']['physical_variable_names']):
            self._phys_idx[label] = i

        if 'charge_scale' in self.parameters['adimensionalization'].keys():
            self._e = self.parameters['adimensionalization']['charge_scale']
        else: 
            self._e = 1.60217662e-19
            self.parameters['adimensionalization']['charge_scale'] = self.e

        if self._n_partitions == 1:
            #no need to recombine meshes
            self._combined_to_full = False
            self._combined_boundary = False
            self._solution_glob = None
            self._gradient_glob = None
            self._magnetic_field_glob = None
            self._magnetic_field_glob_unit = None
            self._jtor_glob = None
            


        
        else:
            self._combined_to_full = False            
            self._combined_boundary = False
            self._solution_glob = None
            self._gradient_glob = None
            self._magnetic_field_glob = None
            self._magnetic_field_unit_glob = None
            self._jtor_glob = None    

        
        
     


    @property
    def parameters(self):
        """Dictionary with solution parameters"""
        return self._parameters

    @property
    def atomic_parameters(self):
        """Dictionary with atomic parameters"""
        return self._atomic_parameters
    @atomic_parameters.setter
    def atomic_parameters(self,value):
        self._atomic_parameters = value

    @property
    def dnn_parameters(self):
        """Dictionary with atomic parameters"""
        return self._dnn_parameters
    @dnn_parameters.setter
    def dnn_parameters(self,value):
        self._dnn_parameters = value
        self._dnn_parameters['dnn_max_adim']=(self._dnn_parameters['dnn_max']/
                                              self.parameters['adimensionalization']['length_scale']**2*
                                              self.parameters['adimensionalization']['time_scale'])
        if not value['const']:
            self._dnn_parameters['dnn_min_adim']=(self._dnn_parameters['dnn_min']/
                                              self.parameters['adimensionalization']['length_scale']**2*
                                              self.parameters['adimensionalization']['time_scale'])
    @property
    def dk_parameters(self):
        """Dictionary with atomic parameters"""
        return self._dk_parameters
    @dk_parameters.setter
    def dk_parameters(self,value):
        self._dk_parameters = value
        self._dk_parameters['dk_max_adim']=(self._dk_parameters['dk_max']/
                                              self.parameters['adimensionalization']['length_scale']**2*
                                              self.parameters['adimensionalization']['time_scale'])
        
        self._dk_parameters['dk_min_adim']=(self._dk_parameters['dk_min']/
                                              self.parameters['adimensionalization']['length_scale']**2*
                                              self.parameters['adimensionalization']['time_scale'])
    @property
    def neq(self):
        """number of equations"""
        return self._neq

    @property
    def ndim(self):
        """number of dimensions"""
        return self._ndim
    
    @property
    def nphys(self):
        """number of physical variables"""
        return self._nphys

    @property
    def ndim(self):
        """number of partitions"""
        return self._ndim

    @property
    def raw_solutions(self):
        """raw soutions on nodes partitions"""
        return self._raw_solutions

    @property
    def raw_solutions_skeleton(self):
        """raw soutions on skeleton on partitions"""
        return self._raw_solutions_skeleton

    @property
    def raw_gradients(self):
        """raw gradients on nodes on partitions"""
        return self._raw_gradients
    
    @property
    def raw_equilibriums(self):
        """raw equilibrium dictionaries on nodes on partitions"""
        return self._raw_equilibriums

    @property
    def raw_solution_boundary_infos(self):
        """raw bounday info dictionaries on nodes on partitions"""
        return self._raw_solution_boundary_infos

    @property
    def mesh(self):
        """mesh on which the solution is calculated"""
        return self._mesh

    @property
    def n_partitions(self):
        """number of partitions"""
        return self._n_partitions

    @property
    def solution_simple(self):
        """solution simply united on a single mesh (means not taking into account repeating points) [Nvertices x neq]"""
        return self._solution_simple
    
    @property
    def gradient_simple(self):
        """gradient simply united on a single mesh (means not taking into account repeating points) [Nvertices x neq x ndim]"""
        return self._gradient_simple
    
    @property
    def solution_simple_phys(self):
        """physical solution simply united on a single mesh (means not taking into account repeating points) [Nvertices x nphys]"""
        return self._solution_simple_phys
    
    @property
    def gradient_simple_phys(self):
        """phisical gradient simply united on a single mesh (means not taking into account repeating points) [Nvertices x nphys x ndim]"""
        return self._gradient_simple_phys

    @property
    def magnetic_field_simple(self):
        """magnetic field recombined united on a single mesh (means not taking into account repeating points) [Nvertices x 3]"""
        return self._magnetic_field_simple

    @property
    def poloidal_flux_simple(self):
        """poloidal flux recombined united on a single mesh (means not taking into account repeating points) [Nvertices x 3]"""
        return self._poloidal_flux_simple
    
    @property
    def solution_boundary(self):
        """conservative solution on faces of the boundary [Nextfaces x n_nodes_per_face x neq]"""
        return self._solution_boundary

    @property
    def solution_skeleton_boundary(self):
        """conservative skeleton solution on faces of the boundary [Nextfaces x n_nodes_per_face x neq]"""
        return self._solution_skeleton_boundary
    
    @property
    def gradient_boundary(self):
        """ gradient  on a boundary [Nextfaces x n_nodes_per_face x neq x ndim]"""
        return self._gradient_boundary
    
    @property
    def magnetic_field_boundary(self):
        """magnetic field recombined on a boundary. This one has shape [Nextfaces x n_nodes_per_face x 3]"""
        return self._magnetic_field_boundary

    @property
    def magnetic_field_unit_boundary(self):
        """magnetic field unit vector recombined on a boundary. This one has shape [Nextfaces x n_nodes_per_face x 3]"""
        return self._magnetic_field_unit_boundary

    @property
    def solution_boundary_gauss(self):
        """conservative solution on gauss points of faces of the boundary [Nextfaces x n_gauss_points_per_face x neq]"""
        return self._solution_boundary_gauss

    @property
    def solution_skeleton_boundary_gauss(self):
        """conservative skeleton solution on gauss points of faces of the boundary [Nextfaces x n_gauss_points_per_face x neq]"""
        return self._solution_skeleton_boundary_gauss
    
    @property
    def gradient_boundary_gauss(self):
        """gradient united on gauss points of faces of the boundary (means not taking into account repeating points) [Nextfaces x n_gauss_points_per_face x neq x ndim]"""
        return self._gradient_boundary_gauss
    
    @property
    def magnetic_field_boundary_gauss(self):
        """magnetic field recombined on gauss points of faces of the boundary. This one has shape [Nextfaces x n_gauss_points_per_face x 3]"""
        return self._magnetic_field_boundary_gauss

    @property
    def magnetic_field_unit_boundary_gauss(self):
        """magnetic field unit vector recombined on gauss points of faces of the boundary. This one has shape [Nextfaces x n_nodes_per_face x 3]"""
        return self._magnetic_field_unit_boundary_gauss

    @property
    def combined_to_full(self):
        """Flag which tells if the solution has been combined to full"""
        return self._combined_to_full
    
    @property
    def combined_boundary(self):
        """Flag which tells if the solution has been combined on a boundary of mesh"""
        return self._combined_boundary

    @property
    def combined_simple_solution(self):
        """Flag which tells if the solution has been combined to simple one on full mesh"""
        return self._combined_simple_solution

    @property
    def full_phys_initialized(self):
        """Flag which tells if physical values has been initialized (full)"""
        return self._full_phys_initialized
    
    @property
    def simple_phys_initialized(self):
        """Flag which tells if physical values has been initialized (simple)"""
        return self._simple_phys_initialized
    @property
    def solution_glob(self):
        """Solution recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x neq]"""
        return self._solution_glob

    @property
    def gradient_glob(self):
        """Gradients recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x neq x ndim]"""
        return self._gradient_glob

    @property
    def magnetic_field_glob(self):
        """magnetic field recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x 3]"""
        return self._magnetic_field_glob

    @property
    def poloidal_flux_glob(self):
        """poloidal flux recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x 3]"""
        return self._poloidal_flux_glob

    @property
    def magnetic_field_unit_glob(self):
        """magnetic field unit vector recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x 3]"""
        return self._magnetic_field_unit_glob

    @property
    def jtor_glob(self):
        """plassma current recombined on a full mesh. This one has shape [Nelems x nodes_per_elem]"""
        return self._jtor_glob

    @property
    def solution_gauss(self):
        """Solution recombined on a full mesh and calculated in gauss points. This one has shape [Nelems x gauss_points_per_elem x neq]"""
        return self._solution_gauss

    @property
    def gradient_gauss(self):
        """Gradients recombined on a full mesh and calculated in gauss points. This one has shape [Nelems x gauss_points_per_elem x neq x ndim]"""
        return self._gradient_gauss

    @property
    def magnetic_field_gauss(self):
        """magnetic field recombined on a full mesh and calculated in gauss points. This one has shape [Nelems x gauss_points_per_elem x 3]"""
        return self._magnetic_field_gauss

    @property
    def magnetic_field_unit_gauss(self):
        """magnetic field recombined on a full mesh and calculated in gauss points. This one has shape [Nelems x gauss_points_per_elem x 3]"""
        return self._magnetic_field_unit_gauss

    @property
    def jtor_gauss(self):
        """plassma current recombined on a full mesh and calculated in gauss points. This one has shape [Nelems x gauss_points_per_elem]"""
        return self._jtor_gauss

    @property
    def solution_glob_phys(self):
        """Physical solution recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x nphys]"""
        return self._solution_glob_phys

    @property
    def gradient_glob_phys(self):
        """Physical gradients recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x nphys x ndim]"""
        return self._gradient_glob_phys
    
    @property
    def e(self):
        """elemental_charge"""
        return self._e

    @property
    def cons_idx(self):
        """dictionary with keys are the cons variables, values are the indexes o the corresponding equation"""
        return self._cons_idx
    
    @property
    def phys_idx(self):
        """dictionary with keys are the phys variables, values are the indexes o the corresponding equation"""
        return self._phys_idx

    @property
    def ionization_source(self):
        """Ionization source on a full solution mesh using conservative values as inputs"""
        return self._ionization_source

    @property
    def ionization_source_simple(self):
        """Ionization source on a simple solution mesh"""
        return self._ionization_source_simple
    
    @property
    def ionization_source_gauss(self):
        """Ionization source on gauss points"""
        return self._ionization_source_gauss

    @property
    def cx_source(self):
        """Charge-exchange source on a full solution mesh using conservative values as inputs"""
        return self._cx_source

    @property
    def cx_source_simple(self):
        """Charge-exchange source on a simple solution mesh"""
        return self._cx_source_simple
    
    @property
    def cx_source_gauss(self):
        """Charge-exchange source on gauss points"""
        return self._cx_source_gauss
    
    @property
    def ohmic_source(self):
        """Ohmic heating source on a full solution mesh using conservative values as inputs"""
        return self._ohmic_source

    @property
    def ohmic_source_simple(self):
        """Ohmic heating source on a simple solution mesh"""
        return self._ohmic_source_simple

    @property
    def ohmic_source_gauss(self):
        """Ohmic heating source on gauss points mesh"""
        return self._ohmic_source_gauss
    @property
    def ionization_rate_simple(self):
        """Ionization rate coefficient on a simple solution mesh"""
        return self._ionization_rate_simple

    @property
    def recombination_rate_simple(self):
        """Recombination rate coefficient on a simple solution mesh"""
        return self._recombination_rate_simple
    
    @property
    def cx_rate_simple(self):
        """Charge exchange rate coefficient on a simple solution mesh"""
        return self._cx_rate_simple

    @property
    def dnn_simple(self):
        """Neutral diffusion on a simple solution mesh"""
        return self._dnn_simple

    @property
    def dnn_simple_with_nn_collision(self):
        """Neutral diffusion with neutral-neutral diffusions on a global solution mesh"""
        return self._dnn_simple_with_nn_collision
    
    @property
    def dnn_simple_with_nn_collision_simple(self):
        """Neutral diffusion with neutral-neutral diffusions on a simple solution mesh"""
        return self._dnn_simple_with_nn_collision_simple

    @property
    def dk_simple(self):
        """Turbulent diffusion on a simple solution mesh"""
        return self._dk_simple
    
    @property
    def dk_glob(self):
        """Turbulent diffusion on a full solution mesh"""
        return self._dk_glob
    
    @property
    def mfp_simple(self):
        """Neutral mean free path on a simple solution mesh"""
        return self._mfp_simple
        
    @property
    def sample_interpolator(self):
        """Sample interpolator for acceleration"""
        return self._sample_interpolator
    @sample_interpolator.setter
    def sample_interpolator(self,value):
        self._sample_interpolator = value
    
    @property
    def solution_interpolators(self):
        """A list of interpolators of solutions in conservative form"""
        return self._solution_interpolators

    @property
    def gradient_interpolators(self):
        """A list of interpolators of solutions in conservative form"""
        return self._gradient_interpolators

    @property
    def field_interpolators(self):
        """A list of interpolators of magnetic field"""
        return self._field_interpolators
    
    @property
    def qcyl_interpolator(self):
        """A list of interpolators of magnetic field"""
        return self._qcyl_interpolator

    @property
    def r_axis(self):
        """R coordinate of magnetic axis"""
        return self._r_axis
    
    @property
    def z_axis(self):
        """Z coordinate of magnetic axis"""
        return self._z_axis

    @property
    def a_glob(self):
        """minor radii on global mesh"""
        return self._a_glob
    
    @property
    def a_simple(self):
        """minor radii on simple mesh"""
        return self._a_simple

    @property
    def qcyl_glob(self):
        """Cylindrical safety factor on global mesh"""
        return self._qcyl_glob
    
    @property
    def qcyl_simple(self):
        """Cylindrical safety factor on simple mesh"""
        return self._qcyl_simple
    



    def recombine_full_solution(self):
        """ 
        Recombine raw solutions into one single mesh
        """
        if not self.mesh.combined_to_full:
            print('Comibining first mesh full')
            self.mesh.recombine_full_mesh()

        self._solution_glob = \
            np.zeros((self.mesh._nelems_glob,self.mesh.mesh_parameters['nodes_per_element'],self.neq))
        self._gradient_glob = \
            np.zeros((self.mesh._nelems_glob,self.mesh.mesh_parameters['nodes_per_element'],self.neq,self.ndim))
        self._magnetic_field_glob = \
            np.zeros((self.mesh._nelems_glob,self.mesh.mesh_parameters['nodes_per_element'],3))
        self._poloidal_flux_glob = \
            np.zeros((self.mesh._nelems_glob,self.mesh.mesh_parameters['nodes_per_element']))
        if self.parameters['switches']['ohmicsrc'][0]==1:
            self._jtor_glob = \
                np.zeros((self.mesh._nelems_glob,self.mesh.mesh_parameters['nodes_per_element']))

        for i in range(self.n_partitions):
            # reshape to the shape of elements
            raw_solution = self.raw_solutions[i].reshape(self.raw_solutions[i].shape[0]//self.mesh.mesh_parameters['nodes_per_element'],self.mesh.mesh_parameters['nodes_per_element'],self.neq)            

            # reshape to the shape of the elements
            raw_gradient = self.raw_gradients[i].reshape(self.raw_gradients[i].shape[0]//self.mesh.mesh_parameters['nodes_per_element'],self.mesh.mesh_parameters['nodes_per_element'],self.neq,self.ndim)            

            #magnetic field
            raw_field = self.raw_equilibriums[i]['magnetic_field'][self.mesh.raw_connectivity[i]] 

            #poloidal flux
            raw_poloidal_flux = self.raw_equilibriums[i]['poloidal_flux'][self.mesh.raw_connectivity[i]] 

            if self.parameters['switches']['ohmicsrc'][0]==1:
                #plasma current
                raw_jtor = self.raw_equilibriums[i]['plasma_current'][self.mesh.raw_connectivity[i]]
            
            if self.n_partitions>1:
                # removing ghost elements
                raw_solution = raw_solution[~self.mesh.raw_ghost_elements[i].astype(bool).flatten(),:]
                self._solution_glob[self.mesh.raw_rest_mesh_data[i]['loc2glob_el'][~self.mesh.raw_ghost_elements[i].flatten()]] = raw_solution

                raw_gradient = raw_gradient[~self.mesh.raw_ghost_elements[i].astype(bool).flatten(),:,:]
                self._gradient_glob[self.mesh.raw_rest_mesh_data[i]['loc2glob_el'][~self.mesh.raw_ghost_elements[i].flatten()]] = raw_gradient

                raw_field = raw_field[~self.mesh.raw_ghost_elements[i].astype(bool).flatten(),:]
                self._magnetic_field_glob[self.mesh.raw_rest_mesh_data[i]['loc2glob_el'][~self.mesh.raw_ghost_elements[i].flatten()]]  = raw_field

                raw_poloidal_flux = raw_poloidal_flux[~self.mesh.raw_ghost_elements[i].astype(bool).flatten()]
                self._poloidal_flux_glob[self.mesh.raw_rest_mesh_data[i]['loc2glob_el'][~self.mesh.raw_ghost_elements[i].flatten()]]  = raw_poloidal_flux

                if self.parameters['switches']['ohmicsrc'][0]==1:
                    raw_jtor = raw_jtor[~self.mesh.raw_ghost_elements[i].astype(bool).flatten(),:]
                    self._jtor_glob[self.mesh.raw_rest_mesh_data[i]['loc2glob_el'][~self.mesh.raw_ghost_elements[i].flatten()]]  = raw_jtor
            else:
                self._solution_glob = raw_solution
                self._gradient_glob = raw_gradient
                self._magnetic_field_glob = raw_field
                self._poloidal_flux_glob = raw_poloidal_flux
                if self.parameters['switches']['ohmicsrc'][0]==1:
                    self._jtor_glob = raw_jtor

        self._magnetic_field_unit_glob = self._magnetic_field_glob/np.sqrt((self._magnetic_field_glob**2).sum(axis=-1))[:,:,None]
        self._combined_to_full = True

    def recombine_simple_full_solution(self):
        """
        Obtain the solution and gradient in a size the same as the vertices
        For the repeating vertices only one (we do not actually now which) value is saved
        This routine is useful for simple overview plots
        """
        if not self.combined_to_full:
            print('Comibining first solution full')
            self.recombine_full_solution()
        self._solution_simple = np.zeros([self.mesh.vertices_glob.shape[0],self.neq])
        #back reshaping
        # here in connectivity matrix there might be multiple entries of the same node. 
        # Therefore the solution in the last entrance of each node last one will be used
        self._solution_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel(),:] = self.solution_glob.reshape(self.solution_glob.shape[0]*self.solution_glob.shape[1],self.neq)

        self._gradient_simple = np.zeros([self.mesh.vertices_glob.shape[0],self.neq,self.ndim])
        self._gradient_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel(),:,:] = self.gradient_glob.reshape(self.gradient_glob.shape[0]*self.gradient_glob.shape[1],self.neq,self.ndim)

        self._magnetic_field_simple = np.zeros([self.mesh.vertices_glob.shape[0],3])
        self._magnetic_field_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel(),:] = self.magnetic_field_glob.reshape(self.magnetic_field_glob.shape[0]*self.magnetic_field_glob.shape[1],3)

        self._poloidal_flux_simple = np.zeros([self.mesh.vertices_glob.shape[0]])
        self._poloidal_flux_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel()] = self.poloidal_flux_glob.reshape(self.poloidal_flux_glob.shape[0]*self.poloidal_flux_glob.shape[1])
        self._combined_simple_solution = True
    
    def recombine_boundary_solution(self):
        """
        extracts solutions and its gradients on the boundary
        """
        if not self.combined_to_full:
            print('Comibining first solution full')
            self.recombine_full_solution()
        if self.mesh.connectivity_b_glob is None:
            print('Comibining first boundary connectivity and info')
            self.mesh.recombine_full_boundary(self.raw_solution_boundary_infos)
        if self.mesh.reference_element is None:
            raise ValueError("Please, provide reference element to the mesh")

        self._solution_boundary = {}
        self._gradient_boundary = {}
        self._magnetic_field_boundary = {}
        self._magnetic_field_unit_boundary = {}
        self._solution_skeleton_boundary = {}

        for key in self.mesh._connectivity_b_glob.keys():
            self._solution_boundary[key] = []
            self._gradient_boundary[key] = []

            self._magnetic_field_boundary[key] = []
            self._magnetic_field_unit_boundary[key] = []
            for face_element_number,face_local_number in zip(self.mesh.face_element_number[key],self.mesh.face_local_number[key]):
                self._solution_boundary[key].append(self.solution_glob[face_element_number,self.mesh.reference_element['faceNodes'][face_local_number,:],:])
                self._gradient_boundary[key].append(self.gradient_glob[face_element_number,self.mesh.reference_element['faceNodes'][face_local_number,:],:,:])

                self._magnetic_field_boundary[key].append(self.magnetic_field_glob[face_element_number,self.mesh.reference_element['faceNodes'][face_local_number,:],:])
                self._magnetic_field_unit_boundary[key].append(self.magnetic_field_unit_glob[face_element_number,self.mesh.reference_element['faceNodes'][face_local_number,:],:])

        #re building skeleton solution
        if self.n_partitions>1:
            solution_skeleton_boundary = np.ones((self.mesh._nfaces_glob,self.mesh.mesh_parameters['nodes_per_face'],self.neq))
            for i in range(self.n_partitions):
                non_ghost = (~self.mesh.raw_ghost_faces[i].flatten())
                raw_solution = self.raw_solutions_skeleton[i].reshape(self.raw_solutions_skeleton[i].shape[0]//self.mesh.mesh_parameters['nodes_per_face'],self.mesh.mesh_parameters['nodes_per_face'],self.neq)
                solution_skeleton_boundary[self.mesh.raw_rest_mesh_data[i]['loc2glob_fa'][:][non_ghost],:] = \
                    raw_solution[non_ghost]
            solution_skeleton_boundary = solution_skeleton_boundary[self.mesh._filled,:,:]
        else:
            solution_skeleton_boundary = self.raw_solutions_skeleton[0].reshape(self.raw_solutions_skeleton[0].shape[0]//self.mesh.mesh_parameters['nodes_per_face'],self.mesh.mesh_parameters['nodes_per_face'],self.neq)
            solution_skeleton_boundary = solution_skeleton_boundary[-self.mesh.raw_mesh_numbers[0]['Nextfaces']:,:,:]
        self._solution_skeleton_boundary = {}
        print(solution_skeleton_boundary.shape)
        for key,indices in self.mesh._indices.items():
            self._solution_skeleton_boundary[key] = []
            for ind in indices:
                self._solution_skeleton_boundary[key].append(solution_skeleton_boundary[ind,:,:])

        self._combined_boundary = True
        
    
    def calculate_in_gauss_points(self):
        if not self.combined_to_full:
            print('Comibining first solution full')
            self.recombine_full_solution()
        if self.mesh.reference_element is None:
            raise ValueError("Please, provide reference element to the mesh")

        self._solution_gauss = np.einsum('ij,kjh->kih', self.mesh.reference_element['N'],self.solution_glob)
        self._gradient_gauss = np.einsum('ij,kjhl->kihl', self.mesh.reference_element['N'],self.gradient_glob)
        self._magnetic_field_gauss = np.einsum('ij,kjh->kih', self.mesh.reference_element['N'],self.magnetic_field_glob)
        self._magnetic_field_unit_gauss = np.einsum('ij,kjh->kih', self.mesh.reference_element['N'],self.magnetic_field_unit_glob)
        self._jtor_gauss = np.einsum('ij,kj->ki', self.mesh.reference_element['N'],self.jtor_glob)
    
    def calculate_in_boundary_gauss_points(self,boundaries):
        if self.mesh.reference_element is None:
            raise ValueError("Please, provide reference element to the mesh")
        if not self.combined_boundary:
            print('Comibining first values on boundary')
            self.recombine_boundary_solution()
        boundary_ordering,connectivity_ordered = self.mesh.calculate_gauss_boundary(boundaries,self.raw_solution_boundary_infos)

        solution_boundary_ordered = np.empty([0,self.solution_boundary[boundaries[0]][0].shape[1],self.solution_boundary[boundaries[0]][0].shape[2]])
        solution_skeleton_boundary_ordered = np.empty([0,self.solution_skeleton_boundary[boundaries[0]][0].shape[1],self.solution_skeleton_boundary[boundaries[0]][0].shape[2]])
        gradient_boundary_ordered = np.empty([0,self._gradient_boundary[boundaries[0]][0].shape[1],self._gradient_boundary[boundaries[0]][0].shape[2],self._gradient_boundary[boundaries[0]][0].shape[3]])
        magnetic_field_boundary_ordered = np.empty([0,self._magnetic_field_boundary[boundaries[0]][0].shape[1],self._magnetic_field_boundary[boundaries[0]][0].shape[2]])
        magnetic_field_unit_boundary_ordered = np.empty([0,self._magnetic_field_unit_boundary[boundaries[0]][0].shape[1],self._magnetic_field_unit_boundary[boundaries[0]][0].shape[2]])
        for bound_ordering in boundary_ordering:
            solution_boundary_ordered = np.vstack([solution_boundary_ordered,self.solution_boundary[boundaries[bound_ordering[0]]][bound_ordering[1]]])
            solution_skeleton_boundary_ordered = np.vstack([solution_skeleton_boundary_ordered,self.solution_skeleton_boundary[boundaries[bound_ordering[0]]][bound_ordering[1]]])
            gradient_boundary_ordered = np.vstack([gradient_boundary_ordered,self._gradient_boundary[boundaries[bound_ordering[0]]][bound_ordering[1]]])
            magnetic_field_boundary_ordered = np.vstack([magnetic_field_boundary_ordered,self._magnetic_field_boundary[boundaries[bound_ordering[0]]][bound_ordering[1]]])
            magnetic_field_unit_boundary_ordered = np.vstack([magnetic_field_unit_boundary_ordered,self._magnetic_field_unit_boundary[boundaries[bound_ordering[0]]][bound_ordering[1]]])

        self._solution_boundary_gauss = np.einsum('ij,kjh->kih', self.mesh.reference_element['N1d'],solution_boundary_ordered)
        self._solution_skeleton_boundary_gauss = np.einsum('ij,kjh->kih', self.mesh.reference_element['N1d'],solution_skeleton_boundary_ordered)
        self._gradient_boundary_gauss = np.einsum('ij,kjhl->kihl', self.mesh.reference_element['N1d'],gradient_boundary_ordered)
        self._magnetic_field_boundary_gauss = np.einsum('ij,kjh->kih', self.mesh.reference_element['N1d'],magnetic_field_boundary_ordered)
        self._magnetic_field_unit_boundary_gauss = np.einsum('ij,kjh->kih', self.mesh.reference_element['N1d'],magnetic_field_unit_boundary_ordered)
        return boundary_ordering,connectivity_ordered

    def summary_along_the_wall(self):
        """
        calculates values in gauss points along the wall
        """

        if self._solution_boundary_gauss is None:
            print('Comibining first values on boundary gauss points')
            self.calculate_in_boundary_gauss_points()

        variables = ['n','u','te','ti','M','p_dyn','gamma_dep','q_dep','recycling','gamma_neut','dl','ds']

        result = {}
        for variable in variables:
            if variable == 'dl':
                res = self.mesh.segment_length_gauss
            elif variable == 'ds':
                res = self.mesh.segment_surface_gauss

            #back-reordering on faces
            result[variable] = res[:,::-1]



        

        



    def plot_overview(self,n_levels=100):
        """
        Plot all conservative variables (dimensional) to have a view on our data
        We also leave the solutions adimensional, providing the dimensional ones as outputs
        """

        if not self._combined_simple_solution:
            print('Comibining first simple solution full')
            self.recombine_simple_full_solution()
        
        solutions_dimensional = self.solution_simple.copy()

        colorbar_labels = []
        
        #in fact for dimensionalization there is field 'reference_values_conservative_variables' but it's not completely correct
        for i in range(self.neq):
            cons_variable =self.parameters['physics']['conservative_variable_names'][i]
            if cons_variable == b'rho':
                solutions_dimensional[:,i]*=self.parameters['adimensionalization']['density_scale']
                colorbar_labels.append(r'n, m$^{-3}$')
                solutions_dimensional[solutions_dimensional[:,i]<1e8,i] = 1e8
            elif cons_variable == b'Gamma':
                solutions_dimensional[:,i]*=self.parameters['adimensionalization']['density_scale']*self.parameters['adimensionalization']['speed_scale']
                colorbar_labels.append(r'$\Gamma$, m$^{-2}$ s$^{-1}$')
            elif cons_variable == b'nEi':
                solutions_dimensional[:,i]*=self.parameters['adimensionalization']['density_scale']*self.parameters['adimensionalization']['specific_energy_scale']
                colorbar_labels.append(r'nE$_i$, m$^{-1}$ s$^{-2}$')
            elif cons_variable == b'nEe':
                solutions_dimensional[:,i]*=self.parameters['adimensionalization']['density_scale']*self.parameters['adimensionalization']['specific_energy_scale']
                colorbar_labels.append(r'nE$_e$, m$^{-1}$ s$^{-2}$')
            elif cons_variable == b'rhon':
                solutions_dimensional[:,i]*=self.parameters['adimensionalization']['density_scale']
                colorbar_labels.append(r'$n_n$, m$^{-3}$')
                solutions_dimensional[solutions_dimensional[:,i]<1e8,i] = 1e8
            elif cons_variable == b'k':
                solutions_dimensional[:,i]*=self.parameters['adimensionalization']['speed_scale']**2
                colorbar_labels.append(r'$k$, m$^{-2}$/s$^{-2}$')
                #solutions_dimensional[solutions_dimensional[:,i]<1e-5,i] = 1e-5
            else:
                raise NameError('Unknown conservative varibale')

        #additional triangulaton: since we have more than 3 points in each element, we can triangulate it
        #take any triangle from the mesh
        if self.mesh.connectivity_big is None:
            self.mesh.create_connectivity_big()
       

        n_lines = int(np.floor(self.neq/2+0.5))
        fig, axes = plt.subplots(n_lines,2, figsize = (15,7.5*n_lines))

        for i in range(self.neq):
            cons_variable =self.parameters['physics']['conservative_variable_names'][i]
            if (cons_variable != b'Gamma') and (cons_variable != b'k') :
                axes[i//2,i%2] = self.mesh.plot_full_mesh(solutions_dimensional[:,i],ax=axes[i//2,i%2],
                                                          log=True,label=colorbar_labels[i],connectivity=self.mesh.connectivity_big,n_levels=n_levels)
            else:
                axes[i//2,i%2] = self.mesh.plot_full_mesh(solutions_dimensional[:,i],ax=axes[i//2,i%2],
                                                          log=False,label=colorbar_labels[i],connectivity=self.mesh.connectivity_big,n_levels=n_levels)
  

        return fig,axes, solutions_dimensional    
        
    def plot_overview_difference(self,second_solution,n_levels=100):
        """
        plots the difference between this and given solution
        """

        if not self._combined_simple_solution:
            print('Comibining first simple solution full')
            self.recombine_simple_full_solution()
        if not second_solution._combined_simple_solution:
            print('Comibining first simple solution of the second one full')
            second_solution.recombine_simple_full_solution()
        
        difference_dimensional = self.solution_simple.copy()-second_solution.solution_simple.copy()
        colorbar_labels = []
        #in fact for dimensionalization there is field 'reference_values_conservative_variables' but it's not completely correct
        for i in range(self.neq):
            cons_variable =self.parameters['physics']['conservative_variable_names'][i]
            if cons_variable == b'rho':
                difference_dimensional[:,i]*=self.parameters['adimensionalization']['density_scale']
                colorbar_labels.append(r'n, m$^{-3}$')
            elif cons_variable == b'Gamma':
                difference_dimensional[:,i]*=self.parameters['adimensionalization']['density_scale']*self.parameters['adimensionalization']['speed_scale']
                colorbar_labels.append(r'$\Gamma$, m$^{-2}$ s$^{-1}$')
            elif cons_variable == b'nEi':
                difference_dimensional[:,i]*=self.parameters['adimensionalization']['density_scale']*self.parameters['adimensionalization']['specific_energy_scale']
                colorbar_labels.append(r'nE$_i$, m$^{-1}$ s$^{-2}$')
            elif cons_variable == b'nEe':
                difference_dimensional[:,i]*=self.parameters['adimensionalization']['density_scale']*self.parameters['adimensionalization']['specific_energy_scale']
                colorbar_labels.append(r'nE$_e$, m$^{-1}$ s$^{-2}$')
            elif cons_variable == b'rhon':
                difference_dimensional[:,i]*=self.parameters['adimensionalization']['density_scale']
                colorbar_labels.append(r'$n_n$, m$^{-3}$')
            elif cons_variable == b'k':
                difference_dimensional[:,i]*=self.parameters['adimensionalization']['speed_scale']**2
                colorbar_labels.append(r'$k$, m$^{-2}$/s$^{-2}$')
                #difference_dimensional[difference_dimensional[:,i]<1e-5,i] = 1e-5
            else:
                raise NameError('Unknown conservative varibale')

        

        #additional triangulaton: since we have more than 3 points in each element, we can triangulate it
        #take any triangle from the mesh
        if self.mesh.connectivity_big is None:
            self.mesh.create_connectivity_big()
       

        n_lines = int(np.floor(self.neq/2+0.5))
        fig, axes = plt.subplots(n_lines,2, figsize = (15,7.5*n_lines))

        for i in range(self.neq):
            cons_variable =self.parameters['physics']['conservative_variable_names'][i]

            axes[i//2,i%2] = self.mesh.plot_full_mesh(difference_dimensional[:,i],ax=axes[i//2,i%2],
                                                          log=False,label=colorbar_labels[i],connectivity=self.mesh.connectivity_big,n_levels=n_levels)
        return fig,axes, difference_dimensional    
    
    def init_phys_variables(self, which='both'):
        ''' 
        converts conservative adymensional SOLEDGE-HDG solutions into physical adimentionalized ones
        :param conservative: u_conservative [n_points x n_equations] for example {n, nu, nEi, nEe} 
                             for 4 equations
        :param which: 'both' -- transfroming both simple and full solutions and gradients
                      'full' -- transfroming only full solutions and gradients
                      'simple' -- transfroming only simple solutions and gradients


        :return physical: physical SOLEDGE-HDG solutions
        for example, for n-Gamma-Ti-Te-neutral model
        u_conservative [n_points x n_equations] for example {n, nu, nEi, nEe,n0} for 4 equations
        u_physical [n_ponts x n_phys_variables] for example {n , u, Ei, Ee, pi, pe, Ti, Te, cs, Mach, n0}
        '''

        
        if which == 'simple':
            if not self.combined_simple_solution:
                print('Comibining first simple solution full')
                self.recombine_simple_full_solution()
            self.cons2phys(self.solution_simple)
            self.cons2phys(self.gradient_simple)
            self._simple_phys_initialized = True
        
        elif which == 'full':
            if not self.combined_to_full:
                print('Comibining first solution full')
                self.recombine_full_solution()
            self.cons2phys(self.solution_glob)
            self.cons2phys(self.gradient_glob)
            self._full_phys_initialized = True

        elif which =='both':
            print('Initializing simple physical solution full')
            self.init_phys_variables(which='simple')
            print('Initializing full physical solution full')
            self.init_phys_variables(which='full')



        
    def cons2phys(self,data):
        """
        converts given solutions or gradients to physical variables with dimensinalizations
        HDG notation:
        U1 = rho_conserv
        U2 = gamma_conserv
        U3 = nEi_conserv
        U4 = nEe_conserv
        U5 = rhon_conserv
        gradinets have "_grad" in the end
        """
        if data.shape[-1] == self.neq:
            #this means that these are soluions
            if len(data.shape) == 3:
                #this means that this is glob solution
                solution_phys = np.zeros((data.shape[0]*data.shape[1],self.nphys))
                data_loc = data.reshape((data.shape[0]*data.shape[1],self.neq))
            elif len(data.shape) == 2:
                #this means that this is simple solution
                solution_phys = np.zeros((data.shape[0],self.nphys))
                data_loc = data.copy()
            for i in range(self.nphys):
                phys_variable =self.parameters['physics']['physical_variable_names'][i]
                if (phys_variable == b'rho'):
                    # n = n_0*U1 (indexing for U in this comments as in fortran)
                    #solution_phys[:,i] = calculate_variable_cons(data_loc,'n',self.parameters['adimensionalization'],self.cons_idx)
                    solution_phys[:,i] = calculate_n_cons(data_loc,self.parameters['adimensionalization']['density_scale'],self.cons_idx)
                elif (phys_variable == b'u'):
                    # u = u_0*U2/U1
                    solution_phys[:,i] = calculate_u_cons(data_loc,self.parameters['adimensionalization']['speed_scale'],self.cons_idx)
                elif (phys_variable == b'Ei'):
                    # Ei = m_0*u_0**2*U3/U1
                    solution_phys[:,i] = calculate_Ei_cons(data_loc,self.parameters['adimensionalization']['speed_scale']**2*self.parameters['adimensionalization']['mass_scale'],self.cons_idx)
                elif (phys_variable == b'Ee'):
                    # Ee = m_0*u_0**2*U4/U1
                    solution_phys[:,i] = calculate_Ee_cons(data_loc,self.parameters['adimensionalization']['speed_scale']**2*self.parameters['adimensionalization']['mass_scale'],self.cons_idx)
                elif (phys_variable == b'pi'):
                    # pi = 2/3/Mref*e*T0*(U3-1/2*U2**2/U1)
                    solution_phys[:,i] = calculate_pi_cons(data_loc,(2/3/self.parameters['physics']['Mref'])* \
                                                          self.parameters['adimensionalization']['density_scale']* \
                                                          self.parameters['adimensionalization']['temperature_scale']* \
                                                          self.parameters['adimensionalization']['charge_scale'] ,self.cons_idx)
                elif (phys_variable == b'pe'):
                    # pe = 2/3/Mref*e*T0*U4
                    solution_phys[:,i] = calculate_pe_cons(data_loc,(2/3/self.parameters['physics']['Mref'])* \
                                                          self.parameters['adimensionalization']['density_scale']* \
                                                          self.parameters['adimensionalization']['temperature_scale']* \
                                                          self.parameters['adimensionalization']['charge_scale'] ,self.cons_idx)
                elif (phys_variable == b'Ti'):
                    # Ti = 2/3/Mref*e*T0*(U3-1/2*U2**2/U1)/U1
                    solution_phys[:,i] = calculate_Ti_cons(data_loc,self.parameters['adimensionalization']['temperature_scale'],self.parameters['physics']['Mref'],self.cons_idx)
                elif (phys_variable == b'Te'):
                    # Te = 2/3/Mref*e*T0*U4/U1
                    solution_phys[:,i] = calculate_Te_cons(data_loc,self.parameters['adimensionalization']['temperature_scale'],self.parameters['physics']['Mref'],self.cons_idx)
                elif (phys_variable == b'Csi'):
                    # cs = u0*(2/3*(U3+U4-1/2*U2**2/U1)/U1)**0.5
                    solution_phys[:,i] = calculate_cs_cons(data_loc,self.parameters['adimensionalization']['speed_scale'],self.cons_idx)
                elif (phys_variable == b'M'):
                    # M = u/cs
                    solution_phys[:,i] = calculate_M_cons(data_loc,self.cons_idx)
                elif (phys_variable == b'rhon'):
                    # n_n = n_0*U5
                    solution_phys[:,i] = calculate_nn_cons(data_loc,self.parameters['adimensionalization']['density_scale'],self.cons_idx)
                elif (phys_variable == b'k'):
                    # k = u_0**2*U6
                    solution_phys[:,i] = calculate_k_cons(data_loc,self.parameters['adimensionalization']['speed_scale']**2,self.cons_idx)
                else:
                    raise KeyError('Unknown variable, go into the code and add this variable if you are sure')

                if len(data.shape) == 3:
                    self._solution_glob_phys = solution_phys.reshape((data.shape[0],data.shape[1],self.nphys))
                elif len(data.shape) == 2:
                    self._solution_simple_phys = solution_phys
                else:
                    raise ValueError('Something weird with the data shape of the solution')

        elif data.shape[-1] == self.ndim:
            # this means that these are the gradients
            if len(data.shape) == 4:
                # this means that this is glob gradients
                grad_phys = np.zeros((data.shape[0]*data.shape[1],self.nphys,self.ndim))
                data_loc = data.reshape((data.shape[0]*data.shape[1],self.neq,self.ndim))
                sol_loc = self.solution_glob.reshape((data.shape[0]*data.shape[1],self.neq))
            elif len(data.shape) == 3:
            #    # this means that this is simple gradients
                grad_phys = np.zeros((data.shape[0],self.nphys,self.ndim))
                data_loc = data.copy()
                sol_loc = self.solution_simple.copy()
            for i in range(self.nphys):
                phys_variable =self.parameters['physics']['physical_variable_names'][i]
                if (phys_variable == b'rho'):
                    # grad(n) = n0/L0*grad(U1)
                    grad_phys[:,i,:] = calculate_grad_n_cons(data_loc,self.parameters['adimensionalization']['density_scale'],
                                                             self.parameters['adimensionalization']['length_scale'],self.cons_idx)
                elif (phys_variable == b'u'):
                    # grad(u) = u0/L0*(-1*grad(U1)*U2/U1**2+grad(U2)/(U1))
                    grad_phys[:,i,:] = calculate_grad_u_cons(sol_loc,data_loc,self.parameters['adimensionalization']['speed_scale'],
                                                             self.parameters['adimensionalization']['length_scale'],self.cons_idx)
                elif (phys_variable == b'Ei'):
                    # grad(Ei) = m_i*u0**2/L0*(-1*grad(U1)*U3/U1**2+grad(U3)/U1)
                    grad_phys[:,i,:] = calculate_grad_Ei_cons(sol_loc,data_loc,(self.parameters['adimensionalization']['speed_scale']**2*
                                                                      self.parameters['adimensionalization']['mass_scale']),
                                                             self.parameters['adimensionalization']['length_scale'],self.cons_idx)
                elif (phys_variable == b'Ee'):
                    # grad(Ee) = m_i*u0**2/L0*(-1*grad(U1)*U4/U1**2+grad(U4)/U1)
                    grad_phys[:,i,:] = calculate_grad_Ee_cons(sol_loc,data_loc,(self.parameters['adimensionalization']['speed_scale']**2*
                                                                      self.parameters['adimensionalization']['mass_scale']),
                                                             self.parameters['adimensionalization']['length_scale'],self.cons_idx)
                elif (phys_variable == b'pi'):
                    # grad(pi) = 2/3/Mref*e*T0/L0*(grad(U3)-grad(U2)*U2/U1+1/2*grad(U1)*U2**2/U1**2)
                    p0 =  (2/3/self.parameters['physics']['Mref'])*self.parameters['adimensionalization']['density_scale']* \
                           self.parameters['adimensionalization']['temperature_scale']*self.parameters['adimensionalization']['charge_scale']

                    grad_phys[:,i,:] = calculate_grad_pi_cons(sol_loc,data_loc,p0,
                                                             self.parameters['adimensionalization']['length_scale'],self.cons_idx)
                elif (phys_variable == b'pe'):
                    # grad(pe) = 2/3/Mref*e*T0/L0*(grad(U4))
                    p0 =  (2/3/self.parameters['physics']['Mref'])*self.parameters['adimensionalization']['density_scale']* \
                           self.parameters['adimensionalization']['temperature_scale']*self.parameters['adimensionalization']['charge_scale']

                    grad_phys[:,i,:] = calculate_grad_pe_cons(data_loc,p0,
                                                             self.parameters['adimensionalization']['length_scale'],self.cons_idx)
                elif (phys_variable == b'Ti'):
                    # assuming that Ei and u already calculated
                    # grad(Ti) = T0/L0*2/3/Mref*(grad(Ei)/m_i/u0**2-grad(u)*u/u0**2)
                    grad_phys[:,i,:] = calculate_grad_Ti_cons(sol_loc,data_loc,self.parameters['adimensionalization']['temperature_scale'],
                                                              self.parameters['physics']['Mref'],self.parameters['adimensionalization']['length_scale'],self.cons_idx)
                elif (phys_variable == b'Te'):
                    # assuming that Ee already calculated
                    # grad(Te) = T0/L0*2/3/Mref*(grad(Ee)/m_i/u0**2)
                    grad_phys[:,i,:] = calculate_grad_Te_cons(sol_loc,data_loc,self.parameters['adimensionalization']['temperature_scale'],
                                                              self.parameters['physics']['Mref'],self.parameters['adimensionalization']['length_scale'],self.cons_idx)
                elif (phys_variable == b'Csi'):
                    # assuming cs already calculated
                    #cs = u0*(2/3*(U3+U4-1/2*U2**2/U1)/U1)**0.5
                    # grad(cs) = u0/L0/2/(cs/u0)*(2/3)*(grad(U1)*(-U3/U1**2-U4/U1**2+U2**2/U1**3)+
                    #                                   grad(U2)*(-U2/U1**2)+grad(U3)/U1+grad(U4)/U1)
                    grad_phys[:,i,:] = calculate_grad_cs_cons(sol_loc,data_loc,self.parameters['adimensionalization']['speed_scale'],
                                                              self.parameters['adimensionalization']['length_scale'],self.cons_idx)
                elif (phys_variable == b'M'):
                    # assuming cs and u already calculated
                    # M = u/cs
                    # grad(M) = grad(u)/cs-grad(cs)*u/cs**2

                    grad_phys[:,i,:] = calculate_grad_M_cons(sol_loc,data_loc,self.parameters['adimensionalization']['length_scale'],self.cons_idx)
                elif (phys_variable == b'rhon'):
                    #grad(n_n) = n0/L0*grad(U5)
                    grad_phys[:,i,:] = calculate_grad_nn_cons(data_loc,self.parameters['adimensionalization']['density_scale'],
                                                             self.parameters['adimensionalization']['length_scale'],self.cons_idx)
                    grad_phys[:,i,:] *= self.parameters['adimensionalization']['density_scale']
                elif (phys_variable == b'k'):
                    #grad(k) = u0**2/L0*grad(U6)
                    grad_phys[:,i,:] = calculate_grad_k_cons(data_loc,self.parameters['adimensionalization']['speed_scale']**2,
                                                              self.parameters['adimensionalization']['length_scale'],self.cons_idx)
                    


            if len(data.shape) == 4:
                    self._gradient_glob_phys = grad_phys.reshape((data.shape[0],data.shape[1],self.nphys,self.ndim))
            elif len(data.shape) == 3:
                self._gradient_simple_phys = grad_phys
                                                    
    def plot_overview_physical(self,n_levels=100, limits=None):
            """
            Plot n, n_n, Ti, Te, M,k,....
            As a physical overview legacy
            """

            if not self._simple_phys_initialized:
                print('Initializing physical solution first')
                self.init_phys_variables('simple')



            colorbar_labels = [r'n, m$^{-3}$',r'$n_n$, m$^{-3}$',r'$T_i$',r'$T_e$',r'M', r'k']
            solutions_plot = np.zeros_like(self.solution_simple)
            solutions_plot[:,0] = self.solution_simple_phys[:,0] #ne
            solutions_plot[:,1] = self.solution_simple_phys[:,-1] #n_n
            if self.neq>2:
                solutions_plot[:,2] = self.solution_simple_phys[:,6] #Ti
                solutions_plot[:,3] = self.solution_simple_phys[:,7] #Te
            if self.neq>4:
                solutions_plot[:,1] = self.solution_simple_phys[:,10] #n_n
            if self.neq>5:
                solutions_plot[:,5] = self.solution_simple_phys[:,11] #k
            solutions_plot[:,4] = self.solution_simple_phys[:,9] #M




            #additional triangulaton: since we have more than 3 points in each element, we can triangulate it
            #take any triangle from the mesh
            if self.mesh.connectivity_big is None:
                self.mesh.create_connectivity_big()


            n_lines = int(np.floor(self.neq/2+0.5))
            fig, axes = plt.subplots(n_lines,2, figsize = (15,7.5*n_lines))

            for i in range(self.neq):
                if limits == None:
                    limit = None
                else:
                    limit = limits[i]
                if ((i!=4)and(i!=5)) :
                    data = solutions_plot[:,i].copy()
                    if (i == 0) or (i == 1):
                        data[data<0] = 1e8
                    else:
                        data[data<0] = 1e-3
                    axes[i//2,i%2] = self.mesh.plot_full_mesh(data,ax=axes[i//2,i%2],
                                                              log=True,label=colorbar_labels[i],connectivity=self.mesh.connectivity_big,n_levels=n_levels,limits=limit)
                else:
                    data = solutions_plot[:,i].copy()
                    data[np.where(np.isnan(data))] = 0
                    axes[i//2,i%2] = self.mesh.plot_full_mesh(data,ax=axes[i//2,i%2],
                                                              log=False,label=colorbar_labels[i],connectivity=self.mesh.connectivity_big,n_levels=n_levels,limits=limit)
    

            return fig,axes, solutions_plot    


    def plot_overview_physical_difference(self,second_solution,n_levels=100):
            """
            Plot difference for n, n_n, Ti, Te, M for this solution and given
            As a physical overview legacy
            """

            if not self._simple_phys_initialized:
                print('Initializing physical solution first')
                self.init_phys_variables('simple')
            if not second_solution._simple_phys_initialized:
                print('Initializing physical solution first')
                second_solution.init_phys_variables('simple')


            colorbar_labels = [r'n, m$^{-3}$',r'$n_n$, m$^{-3}$',r'$T_i$',r'$T_e$',r'M', r'k']
            solutions_plot = np.zeros_like(self.solution_simple)
            solutions_plot[:,0] = self.solution_simple_phys[:,0]-second_solution.solution_simple_phys[:,0] #ne
           
            solutions_plot[:,4] = self.solution_simple_phys[:,9]-second_solution.solution_simple_phys[:,9] #M
            if self.neq>2:
                solutions_plot[:,2] = self.solution_simple_phys[:,6]-second_solution.solution_simple_phys[:,6] #Ti
                solutions_plot[:,3] = self.solution_simple_phys[:,7]-second_solution.solution_simple_phys[:,7] #Te
            if self.neq>4:
                solutions_plot[:,1] = self.solution_simple_phys[:,10]-second_solution.solution_simple_phys[:,10] #n_n
            if self.neq>5:
                solutions_plot[:,5] = self.solution_simple_phys[:,11]-second_solution.solution_simple_phys[:,11] #k




            #additional triangulaton: since we have more than 3 points in each element, we can triangulate it
            #take any triangle from the mesh
            if self.mesh.connectivity_big is None:
                self.mesh.create_connectivity_big()


            n_lines = int(np.floor(self.neq/2+0.5))
            fig, axes = plt.subplots(n_lines,2, figsize = (15,7.5*n_lines))

            for i in range(self.neq):

                axes[i//2,i%2] = self.mesh.plot_full_mesh(solutions_plot[:,i],ax=axes[i//2,i%2],
                                                              log=False,label=colorbar_labels[i],connectivity=self.mesh.connectivity_big,n_levels=n_levels)
    

            return fig,axes,solutions_plot

    def plot_variables_overview(self,variable_list,labels,limits,n_levels,ticks,tick_lables,logs,title=None):
        """
        plots 2D plots of desired varibales
        """
        defined_variables = ['n','nn','te','ti','M','dnn','k','dk']
        for variable in variable_list:
            if variable not in defined_variables:
                raise KeyError(f'{variable} is not in the list of posible variables: {defined_variables}')
        #additional triangulaton: since we have more than 3 points in each element, we can triangulate it
        #take any triangle from the mesh
        if self.mesh.connectivity_big is None:
            self.mesh.create_connectivity_big()

        if not self._combined_simple_solution:
            print('Comibining first simple solution full')
            self.recombine_simple_full_solution()

        #collect dictionary to plot
        var_to_plot = len(variable_list)
        if var_to_plot == 1:
            fig, axes = plt.subplots(1,1, figsize = (7.5,7.5))
        else:
            n_lines = int(np.floor(var_to_plot/2+0.5))
            fig, axes = plt.subplots(n_lines,2, figsize = (15,7.5*n_lines))
        if title is not None:
            fig.suptitle(title)
        res = {}
        for i,(variable,label,limit,tick,tick_label,log) \
            in enumerate(zip(variable_list,labels,limits,ticks,tick_lables,logs)):
            if variable == 'n':
                data = calculate_n_cons(self.solution_simple,self.parameters['adimensionalization']['density_scale'],self.cons_idx)
            elif variable == 'nn':
                data = calculate_nn_cons(self.solution_simple,self.parameters['adimensionalization']['density_scale'],self.cons_idx)
            elif variable == 'te':
                data = calculate_Te_cons(self.solution_simple,self.parameters['adimensionalization']['temperature_scale'],self.parameters['physics']['Mref'],self.cons_idx)
            elif variable == 'ti':
                data = calculate_Ti_cons(self.solution_simple,self.parameters['adimensionalization']['temperature_scale'],self.parameters['physics']['Mref'],self.cons_idx)
            elif variable == 'M':
                data = calculate_M_cons(self.solution_simple,self.cons_idx)
            elif variable == 'dnn':
                data = calculate_dnn_cons(self.solution_simple,self.dnn_parameters,self.atomic_parameters,
                                                                self._e,self.parameters['adimensionalization']['mass_scale'],
                                                                self.parameters['adimensionalization']['temperature_scale'],
                                                                self.parameters['adimensionalization']['density_scale'],
                                                                self.parameters['physics']['Mref'],
                                                                self.parameters['adimensionalization']['length_scale'],
                                                                self.parameters['adimensionalization']['time_scale'])
            elif variable == 'k':
                data = calculate_k_cons(self.solution_simple,self.parameters['adimensionalization']['speed_scale']**2,self.cons_idx)  
            elif variable == 'dk':
                if self.dk_parameters is None:
                    raise ValueError("Please, provide turbulent diffusion settings for the simulation")

                if (self.r_axis is None) or (self.z_axis is None):
                    self.define_magnetic_axis()
                if (self.a_simple is None):
                    self.define_minor_radii(which='simple')
                if (self.qcyl_simple is None):
                    self.define_qcyl(which='simple')
                data = calculate_dk_cons(self.solution_simple,self.dk_parameters,self.qcyl_simple,self.mesh.vertices_glob[:,0]/self.parameters['adimensionalization']['length_scale'],
                                         self.parameters['adimensionalization']['length_scale']**2/self.parameters['adimensionalization']['time_scale'],
                                         self.cons_idx)                               
            data[np.isnan(data)] = limit[0]
            if log:
                data[data<0] = 10.**limit[0]
            res[variable] = data
            if var_to_plot>2:
                axes[i//2,i%2] = self.mesh.plot_full_mesh(data,ax=axes[i//2,i%2],
                             log=log,label=label,connectivity=self.mesh.connectivity_big,n_levels=n_levels,
                             ticks=tick,tick_labels=tick_label,limits=limit)
            elif var_to_plot==2:
                axes[i%2] = self.mesh.plot_full_mesh(data,ax=axes[i%2],
                             log=log,label=label,connectivity=self.mesh.connectivity_big,n_levels=n_levels,
                             ticks=tick,tick_labels=tick_label,limits=limit)
            else:
                axes = self.mesh.plot_full_mesh(data,ax=axes,
                             log=log,label=label,connectivity=self.mesh.connectivity_big,n_levels=n_levels,
                             ticks=tick,tick_labels=tick_label,limits=limit)
        
        return fig,axes,res


    def define_magnetic_axis(self):
        """
        defines magnetic axis as minimum of psi
        """
        if not self.combined_simple_solution:
            print('Comibining first simple solution full')
            self.recombine_simple_full_solution()

        self._r_axis, self._z_axis = self.mesh.vertices_glob[np.where(self.poloidal_flux_simple == self.poloidal_flux_simple.min())][0]
    
    def define_minor_radii(self,which='simple'):
        """
        calculates minor radii either with given magnetic axis
        """
        if which == 'simple':
            if (self._r_axis is None) or (self._z_axis is None):
                self.define_magnetic_axis()
            if (not self.mesh._combined_to_full):
            
                print('Comibining to full mesh')
                self.mesh.recombine_full_mesh()
            self.define_minor_radii(which="full")

            self._a_simple = np.zeros(self.mesh.vertices_glob.shape[0])
            self._a_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel()] = self._a_glob.reshape(self._a_glob.shape[0]*self._a_glob.shape[1])

        if which == 'full':
            if (self._r_axis is None) or (self._z_axis is None):
                self.define_magnetic_axis()
            if (not self.mesh._combined_to_full):
            
                print('Comibining to full mesh')
                self.mesh.recombine_full_mesh()

            #here we should set coordinates of all nodes of all elements
            self._a_glob = calculate_a(self.mesh.vertices_glob[self.mesh.connectivity_glob],self.r_axis,self.z_axis)


    def define_qcyl(self,which='simple'):
        """
        calculates cylindrical safety factor radii either with given magnetic axis
        """
        if which == 'simple':
            if (not self.mesh._combined_to_full):            
                print('Comibining to full mesh')
                self.mesh.recombine_full_mesh()
            if not self.combined_simple_solution:
                print('Comibining first simple solution full')
                self.recombine_simple_full_solution()
            if self.a_simple is None:
                self.define_minor_radii()
            self.define_qcyl(which='full')

            self._qcyl_simple = np.zeros(self.mesh.vertices_glob.shape[0])
            self._qcyl_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel()] = self._qcyl_glob.reshape(self._qcyl_glob.shape[0]*self._qcyl_glob.shape[1])

        elif which == 'full':
            if (not self.mesh._combined_to_full):
            
                print('Comibining to full mesh')
                self.mesh.recombine_full_mesh()
            if self.a_glob is None:
                self.define_minor_radii('full')

            self._qcyl_glob = calculate_q_cyl(self.mesh.vertices_glob[self.mesh.connectivity_glob][:,:,0],self.magnetic_field_glob[:,:,0],
                                             self.magnetic_field_glob[:,:,1],self.magnetic_field_glob[:,:,2],
                                             self.a_glob)




        

    def calculate_variables_along_line(self,r_line,z_line,variable_list):
        """
        calculates plasma parameters noted in variables list on a given line
        returns a dictionary with variables as keys and values along lines for them
        """
        defined_variables = ['n','nn','te','ti','M','dnn','mfp','cx_rate','iz_rate','u',
                             'p_dyn','q_i_par','q_e_par','gamma',
                             'q_i_par_conv','q_i_par_cond',
                             'q_e_par_conv','q_e_par_cond','dk']
        for variable in variable_list:
            if variable not in defined_variables:
                raise KeyError(f'{variable} is not in the list of posible variables: {defined_variables}')
        result = {}
        for variable in variable_list:
            temp = np.zeros_like(z_line)
            if variable == 'n':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.n(r,z)
            elif variable == 'nn':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.nn(r,z)
            elif variable == 'ti':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.ti(r,z)
            elif variable == 'te':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.te(r,z)
            elif variable == 'M':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.M(r,z)
            elif variable == 'dnn':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.dnn(r,z)
            elif variable == 'mfp':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.mfp_nn(r,z)
            elif variable == 'p_dyn':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.p_dyn(r,z)
            elif variable == 'q_i_par':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.ion_heat_flux_par(r,z)
            elif variable == 'q_i_par_conv':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.ion_heat_flux_par_conv(r,z)
            elif variable == 'q_i_par_cond':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.ion_heat_flux_par_cond(r,z)
            elif variable == 'q_e_par':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.electron_heat_flux_par(r,z)
            elif variable == 'q_e_par_conv':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.electron_heat_flux_par_conv(r,z)
            elif variable == 'q_e_par_cond':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.electron_heat_flux_par_cond(r,z)
            elif variable == 'gamma':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.particle_flux_par(r,z)
            elif variable == 'u':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.u(r,z)
            elif variable == 'dk':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.dk(r,z)
            elif variable == 'cx_rate':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.cx_rate(r,z)
            elif variable == 'iz_rate':
                for i,(r,z) in enumerate(zip(r_line,z_line)):
                    temp[i] = self.iz_rate(r,z)
            else:
                raise KeyError(f'{variable} is not in the list of posible variables:  {defined_variables}')
            result[variable] = temp
        return result



    def save_summary_line(self,save_folder,r_line,z_line,variable_list):


        defined_variables = ['n','nn','te','ti','M','dnn','mfp','cx_rate','iz_rate','u',
                             'p_dyn','q_i_par','q_e_par','gamma',
                             'q_i_par_conv','q_i_par_cond',
                             'q_e_par_conv','q_e_par_cond']
        for variable in variable_list:
            if variable not in defined_variables:
                raise KeyError(f'{variable} is not in the list of posible variables: {defined_variables}')
        vertices = np.stack([r_line,z_line]).T
        np.save(f'{save_folder}vertices.npy',vertices)
        
        
        values_on_line = self.calculate_variables_along_line(r_line,z_line,variable_list)
        
        for variable,values in values_on_line.items():
            if variable == 'n':
                np.save(f'{save_folder}density.npy',values)
            elif variable == 'nn':
                np.save(f'{save_folder}neitral_density.npy',values)
            elif variable == 'te':
                np.save(f'{save_folder}te.npy',values)
            elif variable == 'ti':
                np.save(f'{save_folder}ti.npy',values)
            elif variable == 'M':                
                np.save(f'{save_folder}M.npy',values)
            elif variable == 'dnn':
                np.save(f'{save_folder}dnn.npy',values)
            elif variable == 'mfp':
                np.save(f'{save_folder}mfp.npy',values)
            elif variable == 'u':
                np.save(f'{save_folder}u.npy',values)
            elif variable == 'p_dyn':
                np.save(f'{save_folder}p_dyn.npy',values)
            elif variable == 'q_i_par':
                np.save(f'{save_folder}q_i_par.npy',values)
            elif variable == 'q_i_par_conv':
                np.save(f'{save_folder}q_i_par_conv.npy',values)
            elif variable == 'q_i_par_cond':
                np.save(f'{save_folder}q_i_par_cond.npy',values)
            elif variable == 'q_e_par':
                np.save(f'{save_folder}q_e_par.npy',values)
            elif variable == 'q_e_par_conv':
                np.save(f'{save_folder}q_e_par_conv.npy',values)
            elif variable == 'q_e_par_cond':
                np.save(f'{save_folder}q_e_par_cond.npy',values)
            elif variable == 'gamma':
                np.save(f'{save_folder}gamma.npy',values)
            elif variable == 'u':
                np.save(f'{save_folder}u.npy',values)
            elif variable == 'cx_rate':
                np.save(f'{save_folder}cx_rate.npy',values)
            elif variable == 'iz_rate':
                np.save(f'{save_folder}iz_rate.npy',values)
            else:
                raise KeyError(f'{variable} is not in the list of posible variables')

        return values_on_line

    



    def calculate_ohmic_source(self, which='simple'):
        """
            calculate the ionization rate
        """

        if 'ohmic_coeff' not in  self.parameters['physics'].keys():
            raise KeyError('Please, provide ohmic heating adimensionalized coefficient to self.parameters["physics"]')
        if 'Zeff' not in  self.parameters['physics'].keys():
            raise KeyError('Please, effective charge to self.parameters["physics"]')
        
        if which=="simple":
            self.calculate_ohmic_source(which="full")

            self._ohmic_source_simple_simple = np.zeros(self.mesh.vertices_glob.shape[0])
            self._ohmic_source_simple_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel()] = self._ohmic_source.reshape(self.solution_glob.shape[0]*self.solution_glob.shape[1])
        elif which == 'full':
            if not self._combined_to_full:
                self.recombine_full_solution()
            self._ohmic_source = calculate_ohmic_source_cons(self.solution_glob,self.jtor_glob,
                                                            self.parameters['physics']['Mref'],
                                                            self.parameters['adimensionalization']['mass_scale'],
                                                            self.parameters['adimensionalization']['density_scale'],
                                                            self.parameters['adimensionalization']['length_scale'],
                                                            self.parameters['adimensionalization']['time_scale'],
                                                            self.parameters['physics']['ohmic_coeff'],
                                                            self.parameters['physics']['Zeff'])
        elif which == 'gauss':
            if not self._combined_to_full:
                self.recombine_full_solution()
            if self._jtor_gauss is None:
                print('Calculating on gauss points first')
                self.calculate_in_gauss_points()
            self._ohmic_source_gauss = calculate_ohmic_source_cons(self.solution_gauss,self.jtor_gauss,
                                                            self.parameters['physics']['Mref'],
                                                            self.parameters['adimensionalization']['mass_scale'],
                                                            self.parameters['adimensionalization']['density_scale'],
                                                            self.parameters['adimensionalization']['length_scale'],
                                                            self.parameters['adimensionalization']['time_scale'],
                                                            self.parameters['physics']['ohmic_coeff'],
                                                            self.parameters['physics']['Zeff'])
            
            


        



    def calculate_ionization_rate(self,which="simple"):
        """
            calculate the ionization rate
            simple: for simple mesh solution
            full: on full mesh solution
            coordinates: on a line with provided coordinates (to be done)
            gauss_points: on gauss points (to be done)
        """    

        if which=="simple":
            if self.atomic_parameters is None:
                raise ValueError("Please, provide atomic settings for the simulation")
            if "iz" not in self.atomic_parameters.keys():
                raise ValueError("Please, provide ionization atomic settings for the simulation")
            if not self._simple_phys_initialized:
                print('Initializing physical solution first')
                self.init_phys_variables('simple')
            
            self.calculate_ionization_rate(which="full")

            self._ionization_rate_simple = np.zeros(self.mesh.vertices_glob.shape[0])
            self._ionization_rate_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel()] = self._ionization_rate.reshape(self.solution_glob.shape[0]*self.solution_glob.shape[1])
            
        elif which =="full":
            if not self._combined_to_full:
                self.recombine_full_solution()
            self._ionization_rate = calculate_iz_rate_cons(self.solution_glob,self.atomic_parameters['iz'],
                                                                self.parameters['adimensionalization']['temperature_scale'],
                                                                self.parameters['adimensionalization']['density_scale'],
                                                                self.parameters['physics']['Mref'])

    def calculate_recombination_rate(self,which="simple"):
        """
            calculate the recombination rate
            simple: for simple mesh solution
            full: on full mesh solution
            coordinates: on a line with provided coordinates (to be done)
            gauss_points: on gauss points (to be done)
        """    

        if which=="simple":
            if self.atomic_parameters is None:
                raise ValueError("Please, provide atomic settings for the simulation")
            if "iz" not in self.atomic_parameters.keys():
                raise ValueError("Please, provide ionization atomic settings for the simulation")
            if not self._simple_phys_initialized:
                print('Initializing physical solution first')
                self.init_phys_variables('simple')
            
            self.calculate_recombination_rate(which="full")

            self._recombination_rate_simple = np.zeros(self.mesh.vertices_glob.shape[0])
            self._recombination_rate_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel()] = self._recombination_rate.reshape(self.solution_glob.shape[0]*self.solution_glob.shape[1])
            
        elif which =="full":
            if not self._combined_to_full:
                self.recombine_full_solution()
            self._recombination_rate = calculate_rec_rate_cons(self.solution_glob,self.atomic_parameters['rec'],
                                                                self.parameters['adimensionalization']['temperature_scale'],
                                                                self.parameters['adimensionalization']['density_scale'],
                                                                self.parameters['physics']['Mref'])

    def calculate_cx_rate(self,which="simple"):
        """
            calculate the charge exchange rate
            simple: for simple mesh solution
            full: on full mesh solution
            coordinates: on a line with provided coordinates (to be done)
        """    

        if which=="simple":
            if self.atomic_parameters is None:
                raise ValueError("Please, provide atomic settings for the simulation")
            if "iz" not in self.atomic_parameters.keys():
                raise ValueError("Please, provide ionization atomic settings for the simulation")
            if not self._simple_phys_initialized:
                print('Initializing physical solution first')
                self.init_phys_variables('simple')
            
            self.calculate_cx_rate(which="full")

            self._cx_rate_simple = np.zeros(self.mesh.vertices_glob.shape[0])
            self._cx_rate_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel()] = self._cx_rate.reshape(self.solution_glob.shape[0]*self.solution_glob.shape[1])
            
        if which =="full":
            if not self._combined_to_full:
                self.recombine_full_solution()
            self._cx_rate = calculate_cx_rate_cons(self.solution_glob,self.atomic_parameters['cx'],
                                                                self.parameters['adimensionalization']['temperature_scale'],
                                                                self.parameters['physics']['Mref'])

    def calculate_dnn(self,which="simple"):
        """
            calculate neutral diffusion
            simple: for simple mesh solution
            full: on full mesh solution
            coordinates: on a line with provided coordinates (to be done)
        """    

        if which=="simple":
            if self.atomic_parameters is None:
                raise ValueError("Please, provide atomic settings for the simulation")
            if self.dnn_parameters is None:
                raise ValueError("Please, provide neutral diffusion settings for the simulation")
            if "iz" not in self.atomic_parameters.keys():
                raise ValueError("Please, provide ionization atomic settings for the simulation")
            if "cx" not in self.atomic_parameters.keys():
                raise ValueError("Please, provide ionization atomic settings for the simulation")
            if not self._simple_phys_initialized:
                print('Initializing physical solution first')
                self.init_phys_variables('simple')
            
            self.calculate_dnn(which="full")

            self._dnn_simple = np.zeros(self.mesh.vertices_glob.shape[0])
            self._dnn_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel()] = self._dnn.reshape(self.solution_glob.shape[0]*self.solution_glob.shape[1])
            
        if which =="full":
            if not self._combined_to_full:
                self.recombine_full_solution()
            self._dnn = calculate_dnn_cons(self.solution_glob,self.dnn_parameters,self.atomic_parameters,
                                                                self._e,self.parameters['adimensionalization']['mass_scale'],
                                                                self.parameters['adimensionalization']['temperature_scale'],
                                                                self.parameters['adimensionalization']['density_scale'],
                                                                self.parameters['physics']['Mref'],
                                                                self.parameters['adimensionalization']['length_scale'],
                                                                self.parameters['adimensionalization']['time_scale'])

    def calculate_dnn_with_nn_collision(self,which="simple"):
        """
            calculate neutral diffusion with neutral-neutral collisions
            simple: for simple mesh solution
            full: on full mesh solution
            coordinates: on a line with provided coordinates (to be done)
        """    

        if which=="simple":
            if self.atomic_parameters is None:
                raise ValueError("Please, provide atomic settings for the simulation")
            if self.dnn_parameters is None:
                raise ValueError("Please, provide neutral diffusion settings for the simulation")
            if "iz" not in self.atomic_parameters.keys():
                raise ValueError("Please, provide ionization atomic settings for the simulation")
            if "cx" not in self.atomic_parameters.keys():
                raise ValueError("Please, provide ionization atomic settings for the simulation")
            if not self._simple_phys_initialized:
                print('Initializing physical solution first')
                self.init_phys_variables('simple')
            
            self.calculate_dnn_with_nn_collision(which="full")

            self._dnn_with_nn_collision_simple = np.zeros(self.mesh.vertices_glob.shape[0])
            self._dnn_with_nn_collision_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel()] = self._dnn_with_nn_collision.reshape(self.solution_glob.shape[0]*self.solution_glob.shape[1])
            
        if which =="full":
            if not self._combined_to_full:
                self.recombine_full_solution()
            self._dnn_with_nn_collision = calculate_dnn_with_nn_collision_cons(self.solution_glob,self.dnn_parameters,self.atomic_parameters,
                                                                self._e,self.parameters['adimensionalization']['mass_scale'],
                                                                self.parameters['adimensionalization']['temperature_scale'],
                                                                self.parameters['adimensionalization']['density_scale'],
                                                                self.parameters['physics']['Mref'],
                                                                self.parameters['adimensionalization']['length_scale'],
                                                                self.parameters['adimensionalization']['time_scale'])

    def calculate_dk(self,which="simple"):
        """
            calculate neutral diffusion
            simple: for simple mesh solution
            full: on full mesh solution
            coordinates: on a line with provided coordinates (to be done)

        """    

        if which=="simple":
            if self.dk_parameters is None:
                raise ValueError("Please, provide neutral diffusion settings for the simulation")

            if not self._combined_simple_solution:
                print('Initializing physical solution first')
                self.recombine_simple_full_solution()
            if (self.r_axis is None) or (self.z_axis is None):
                    self.define_magnetic_axis()
            if (self.a_simple is None):
                self.define_minor_radii(which='simple')
            if (self.qcyl_simple is None):
                self.define_qcyl(which='simple')
            self.calculate_dk(which='full')
            self._dk_simple = np.zeros(self.mesh.vertices_glob.shape[0])
            self._dk_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel()] = self._dk_glob.reshape(self._dk_glob.shape[0]*self._dk_glob.shape[1])
            
        if which =="full":
            if self.dk_parameters is None:
                raise ValueError("Please, provide neutral diffusion settings for the simulation")

            if not self._combined_simple_solution:
                print('Initializing physical solution first')
                self.recombine_simple_full_solution()
            if (self.r_axis is None) or (self.z_axis is None):
                    self.define_magnetic_axis()
            if (self.a_glob is None):
                self.define_minor_radii(which='full')
            if (self.qcyl_glob is None):
                self.define_qcyl(which='full')

            self._dk_glob = calculate_dk_cons(self.solution_glob,self.dk_parameters,self.qcyl_glob,self.mesh.vertices_glob[self.mesh.connectivity_glob][:,:,0]/self.parameters['adimensionalization']['length_scale'],
                                                self.parameters['adimensionalization']['length_scale']**2/self.parameters['adimensionalization']['time_scale'],
                                                self.cons_idx)

    
    def calculate_mfp(self,which="simple"):
        """
            calculate neutral mean free path
            simple: for simple mesh solution
            full: on full mesh solution
            coordinates: on a line with provided coordinates (to be done)
        """    
        if self.atomic_parameters is None:
                raise ValueError("Please, provide atomic settings for the simulation")
        if self.dnn_parameters is None:
            raise ValueError("Please, provide neutral diffusion settings for the simulation")
        if "iz" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide ionization atomic settings for the simulation")
        if "cx" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide ionization atomic settings for the simulation")
        if not self._simple_phys_initialized:
            print('Initializing physical solution first')
            self.init_phys_variables('both')
        if which=="simple":
            if not self._simple_phys_initialized:
                print('Initializing physical solution first')
                self.init_phys_variables('simple')
                        
            self.calculate_mfp(which="full")

            self._mfp_simple = np.zeros(self.mesh.vertices_glob.shape[0])
            self._mfp_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel()] = self._mfp.reshape(self.solution_glob.shape[0]*self.solution_glob.shape[1])
            
        if which =="full":
            if not self._combined_to_full:
                self.recombine_full_solution()
            if not self._simple_phys_initialized:
                print('Initializing physical solution first')
                self.init_phys_variables('full')
            if self._dnn is None:
                self.calculate_dnn('full')
            self._mfp = calculate_mfp_cons(self.solution_glob,self.dnn_parameters,self.atomic_parameters,
                                                                self._e,self.parameters['adimensionalization']['mass_scale'],
                                                                self.parameters['adimensionalization']['temperature_scale'],
                                                                self.parameters['adimensionalization']['density_scale'],
                                                                self.parameters['physics']['Mref'],
                                                                self.parameters['adimensionalization']['length_scale'],
                                                                self.parameters['adimensionalization']['time_scale'])


    def calculate_ionization_source(self,which="simple"):
        """
            calculate the ionization source
            simple: for simple mesh solution
            full: on full mesh solution (to be done)
            coordinates: on a line with provided coordinates (to be done)
            gauss: on gauss points
        """    
        if self.atomic_parameters is None:
            raise ValueError("Please, provide atomic settings for the simulation")
        if "iz" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide ionization atomic settings for the simulation")
        if which=="simple":
            
            if not self._simple_phys_initialized:
                print('Initializing physical solution first')
                self.init_phys_variables('simple')
            
            self.calculate_ionization_source(which="full")

            self._ionization_source_simple = np.zeros(self.mesh.vertices_glob.shape[0])
            self._ionization_source_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel()] = self._ionization_source.reshape(self.solution_glob.shape[0]*self.solution_glob.shape[1])
            
        if which =="full":
            if not self._combined_to_full:
                self.recombine_full_solution()
            self._ionization_source = calculate_iz_source_cons(self.solution_glob,self.atomic_parameters['iz'],
                                                                self.parameters['adimensionalization']['temperature_scale'],
                                                                self.parameters['adimensionalization']['density_scale'],
                                                                self.parameters['physics']['Mref'])

        if which == 'gauss':
            if self.solution_gauss is None:
                print('Initializing values in gauss points first')
                self.calculate_in_gauss_points()
            self._ionization_source_gauss = calculate_iz_source_cons(self.solution_gauss,self.atomic_parameters['iz'],
                                                                self.parameters['adimensionalization']['temperature_scale'],
                                                                self.parameters['adimensionalization']['density_scale'],
                                                                self.parameters['physics']['Mref'])

    def calculate_cx_source(self,which="simple"):
        """
            calculate the charge-exchange source
            simple: for simple mesh solution
            full: on full mesh solution (to be done)
            coordinates: on a line with provided coordinates (to be done)
            gauss: on gauss points
        """    
        if self.atomic_parameters is None:
            raise ValueError("Please, provide atomic settings for the simulation")
        if "iz" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide ionization atomic settings for the simulation")
        if which=="simple":
            
            if not self._simple_phys_initialized:
                print('Initializing physical solution first')
                self.init_phys_variables('simple')
            
            self.calculate_cx_source(which="full")

            self._cx_source_simple = np.zeros(self.mesh.vertices_glob.shape[0])
            self._cx_source_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel()] = self._cx_source.reshape(self.solution_glob.shape[0]*self.solution_glob.shape[1])
            
        if which =="full":
            if not self._combined_to_full:
                self.recombine_full_solution()
            self._cx_source = calculate_cx_source_cons(self.solution_glob,self.atomic_parameters['cx'],
                                                                self.parameters['adimensionalization']['temperature_scale'],
                                                                self.parameters['adimensionalization']['density_scale'],
                                                                self.parameters['physics']['Mref'])

        if which == 'gauss':
            if self.solution_gauss is None:
                print('Initializing values in gauss points first')
                self.calculate_in_gauss_points()
            self._cx_source_gauss = calculate_cx_source_cons(self.solution_gauss,self.atomic_parameters['cx'],
                                                                self.parameters['adimensionalization']['temperature_scale'],
                                                                self.parameters['adimensionalization']['density_scale'],
                                                                self.parameters['physics']['Mref'])
    def define_interpolators(self):
        """
        defines interpolators for full solutions and gradients based on shape functions
        """

        if not self._combined_simple_solution:
            print('Comibining first simple solution full')
            self.recombine_simple_full_solution()
        if self.mesh.connectivity_big is None:
            print('Comibining first big connectivity')
            self.mesh.create_connectivity_big()
        
        if self.mesh.reference_element is None:
            raise ValueError("Please, provide reference element")
        if self.mesh.element_number is None:
            print('Defining an element number mask')
            self.mesh.make_element_number_funtion()
        if self._qcyl_glob is None:
            self.define_qcyl(which='full')
        if self._sample_interpolator is None:
            if self.mesh.mesh_parameters['element_type'] == 'triangle':
                self._sample_interpolator = SoledgeHDG2DInterpolator(self.mesh.vertices_glob,np.ones_like(self.solution_glob[:,:,0]),self.mesh.connectivity_glob,
                    self.mesh.element_number,self.mesh.reference_element['NodesCoord'],self.mesh.mesh_parameters['element_type'], self.mesh.p_order,limit=False)
            elif self.mesh.mesh_parameters['element_type'] == 'quadrilateral':
                self._sample_interpolator = SoledgeHDG2DInterpolator(self.mesh.vertices_glob,np.ones_like(self.solution_glob[:,:,0]),self.mesh.connectivity_glob,
                    self.mesh.element_number,self.mesh.reference_element['NodesCoord1d'],self.mesh.mesh_parameters['element_type'], self.mesh.p_order,limit=False)
            
        self._solution_interpolators = []
        self._gradient_interpolators = []
        for i in range(self.neq):
            self._solution_interpolators.append(SoledgeHDG2DInterpolator.instance(self._sample_interpolator,self.solution_glob[:,:,i]))
            grad = []
            grad.append(SoledgeHDG2DInterpolator.instance(self._sample_interpolator,self.gradient_glob[:,:,i,0]))
            grad.append(SoledgeHDG2DInterpolator.instance(self._sample_interpolator,self.gradient_glob[:,:,i,1]))
            self._gradient_interpolators.append(grad)

        # magnetic field
        self._field_interpolators = []
        for i in range(3):
            self._field_interpolators.append(SoledgeHDG2DInterpolator.instance(self._sample_interpolator,self.magnetic_field_glob[:,:,i]))

        # q_cylindrical
        self._qcyl_interpolator = SoledgeHDG2DInterpolator.instance(self._sample_interpolator,self.qcyl_glob)

    def n(self,r,z):
        """
        returns value of density in given point (r,z)
        """
        if b'rho' not in self.parameters['physics']['physical_variable_names']:
            raise KeyError('density is not in the models')
        
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        #only fill needed field
        solution = np.zeros([1,self.neq])
        solution[:,self._cons_idx[b'rho']] = self._solution_interpolators[self._cons_idx[b'rho']](r,z)

        return calculate_n_cons(solution,self.parameters['adimensionalization']['density_scale'],self._cons_idx)

    def ti(self,r,z):
        """
        returns value of ion temperature in given point (r,z)
        """
        if b'Ti' not in self.parameters['physics']['physical_variable_names']:
            raise KeyError('ion temperature is not in the models')
        
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        # Ti = T0*2/3/Mref*(U3/U1-1/2*U2**2/U1**2)
        #only fill needed field
        solution = np.zeros([1,self.neq])
        solution[:,self._cons_idx[b'rho']] = self._solution_interpolators[self._cons_idx[b'rho']](r,z)
        if solution[:,self._cons_idx[b'rho']] == 0:
            return 0
        solution[:,self._cons_idx[b'Gamma']] = self._solution_interpolators[self._cons_idx[b'Gamma']](r,z)
        solution[:,self._cons_idx[b'nEi']] = self._solution_interpolators[self._cons_idx[b'nEi']](r,z)
        return calculate_Ti_cons(solution,self.parameters['adimensionalization']['temperature_scale'],
                                 self.parameters['physics']['Mref'],self._cons_idx)

    def te(self,r,z):
        """
        returns value of electron temperature in given point (r,z)
        """
        if b'Te' not in self.parameters['physics']['physical_variable_names']:
            raise KeyError('electron temperature is not in the models')
        
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        # Te = T0*2/3/Mref*(U4/U1)
        #only fill needed field
        solution = np.zeros([1,self.neq])
        solution[:,self._cons_idx[b'rho']] = self._solution_interpolators[self._cons_idx[b'rho']](r,z)
        if solution[:,self._cons_idx[b'rho']] == 0:
            return 0

        solution[:,self._cons_idx[b'nEe']] = self._solution_interpolators[self._cons_idx[b'nEe']](r,z)
        return calculate_Te_cons(solution,self.parameters['adimensionalization']['temperature_scale'],
                                 self.parameters['physics']['Mref'],self._cons_idx)
    
    def u(self,r,z):
        """
        returns value of plasma velocity in given point (r,z)
        """
        if b'u' not in self.parameters['physics']['physical_variable_names']:
            raise KeyError('Mach number is not in the models')
        
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        # u = u0*U2/U1
        #only fill needed field
        solution = np.zeros([1,self.neq])
        solution[:,self._cons_idx[b'rho']] = self._solution_interpolators[self._cons_idx[b'rho']](r,z)
        if solution[:,self._cons_idx[b'rho']] == 0:
            return 0
        solution[:,self._cons_idx[b'Gamma']] = self._solution_interpolators[self._cons_idx[b'Gamma']](r,z)
        return calculate_u_cons(solution,self.parameters['adimensionalization']['speed_scale'],self._cons_idx)
    
    def cs(self,r,z):
        """
        returns value of plasma sound speed in given point (r,z)
        """
        if b'Csi' not in self.parameters['physics']['physical_variable_names']:
            raise KeyError('Mach number is not in the models')
        
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        # cs = u0*(2/3*(U3+U4-1/2*U2**2/U1)/U1)**0.5
        #only fill needed field
        solution = np.zeros([1,self.neq])
        solution[:,self._cons_idx[b'rho']] = self._solution_interpolators[self._cons_idx[b'rho']](r,z)
        if solution[:,self._cons_idx[b'rho']] == 0:
            return 0
        solution[:,self._cons_idx[b'Gamma']] = self._solution_interpolators[self._cons_idx[b'Gamma']](r,z)
        solution[:,self._cons_idx[b'nEi']] = self._solution_interpolators[self._cons_idx[b'nEi']](r,z)
        solution[:,self._cons_idx[b'nEe']] = self._solution_interpolators[self._cons_idx[b'nEe']](r,z)
        return calculate_cs_cons(solution,self.parameters['adimensionalization']['speed_scale'],self._cons_idx)
    
    def M(self,r,z):
        """
        returns value of mach number in given point (r,z)
        """
        if b'M' not in self.parameters['physics']['physical_variable_names']:
            raise KeyError('Mach number is not in the models')
        
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        # M = u/cs
        #only fill needed field
        solution = np.zeros([1,self.neq])
        solution[:,self._cons_idx[b'rho']] = self._solution_interpolators[self._cons_idx[b'rho']](r,z)
        if solution[:,self._cons_idx[b'rho']] == 0:
            return 0
        solution[:,self._cons_idx[b'Gamma']] = self._solution_interpolators[self._cons_idx[b'Gamma']](r,z)
        solution[:,self._cons_idx[b'nEi']] = self._solution_interpolators[self._cons_idx[b'nEi']](r,z)
        solution[:,self._cons_idx[b'nEe']] = self._solution_interpolators[self._cons_idx[b'nEe']](r,z)
        return calculate_M_cons(solution,self._cons_idx)

    def nn(self,r,z):
        """
        returns value of neutral density in given point (r,z)
        """
        if b'rhon' not in self.parameters['physics']['physical_variable_names']:
            raise KeyError('neutral density number is not in the models')
        
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        # nn=n0*U5
        #only fill needed field
        solution = np.zeros([1,self.neq])
        solution[:,self._cons_idx[b'rhon']] = self._solution_interpolators[self._cons_idx[b'rhon']](r,z)
        return calculate_nn_cons(solution,self.parameters['adimensionalization']['density_scale'],self._cons_idx)

    def ionization_source_interp(self,r,z):
        """
        returns ionization source value in given point (r,z)
        """
        if self.atomic_parameters is None:
                raise ValueError("Please, provide atomic settings for the simulation")
        if "iz" not in self.atomic_parameters.keys():
                raise ValueError("Please, provide ionization atomic settings for the simulation")

        
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        solution = np.zeros([1,self.neq])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] ==0:
            return 0


        return calculate_iz_source_cons(solution,self.atomic_parameters['iz'],
                                                 self.parameters['adimensionalization']['temperature_scale'],
                                                 self.parameters['adimensionalization']['density_scale'],
                                                 self.parameters['physics']['Mref'])

    def iz_rate(self,r,z):
        """
        returns value of ionization rate in given point (r,z)
        """
        if self.atomic_parameters is None:
                raise ValueError("Please, provide atomic settings for the simulation")
        if "iz" not in self.atomic_parameters.keys():
                raise ValueError("Please, provide ionization atomic settings for the simulation")
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        
        solution = np.zeros([1,self.neq])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] ==0:
            return 0
        return calculate_iz_rate_cons(solution,self.atomic_parameters['iz'],
                                                 self.parameters['adimensionalization']['temperature_scale'],
                                                 self.parameters['adimensionalization']['density_scale'],
                                                 self.parameters['physics']['Mref'])

    def cx_rate(self,r,z):
        """
        returns value of cx rate in given point (r,z)
        """
        if self.atomic_parameters is None:
                raise ValueError("Please, provide atomic settings for the simulation")
        if "cx" not in self.atomic_parameters.keys():
                raise ValueError("Please, provide charge exchange atomic settings for the simulation")
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        
        solution = np.zeros([1,self.neq])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] ==0:
            return 0
        return calculate_cx_rate_cons(solution,self.atomic_parameters['cx'],
                                                 self.parameters['adimensionalization']['temperature_scale'],
                                                 self.parameters['physics']['Mref'])
    
    def dnn(self,r,z):
        """
        returns value of neutral diffusion in given point (r,z)
        """
        if self.atomic_parameters is None:
                raise ValueError("Please, provide atomic settings for the simulation")
        if "cx" not in self.atomic_parameters.keys():
                raise ValueError("Please, provide charge exchange atomic settings for the simulation")
        if "iz" not in self.atomic_parameters.keys():
                raise ValueError("Please, provide ionization atomic settings for the simulation")
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        
        solution = np.zeros([1,self.neq])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] ==0:
            return 0
        return calculate_dnn_cons(solution,self.dnn_parameters, self.atomic_parameters,
                                                                self._e,self.parameters['adimensionalization']['mass_scale'],
                                                                self.parameters['adimensionalization']['temperature_scale'],
                                                                self.parameters['adimensionalization']['density_scale'],
                                                                self.parameters['physics']['Mref'],
                                                                self.parameters['adimensionalization']['length_scale'],
                                                                self.parameters['adimensionalization']['time_scale'])

    def dk(self,r,z):
        """
        returns value of turbulent diffusion in given point (r,z)
        """
        if self.dk_parameters is None:
                raise ValueError("Please, provide turbulent diffusion settings for the simulation")

        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        if self.r_axis is None:
            self.define_minor_radii()
        
        
        solution = np.zeros([1,self.neq])

        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] ==0:
            return 0
        a = np.sqrt((r-self.r_axis)**2+(z-self.z_axis)**2)
        Br = self._field_interpolators[0](r,z)
        Bz = self._field_interpolators[1](r,z)
        Bt = self._field_interpolators[2](r,z)
        q_cyl = calculate_q_cyl(r,Br,Bz,Bt,a)

        return calculate_dk_cons(solution,self.dk_parameters, q_cyl,r,
                                                                self.parameters['adimensionalization']['length_scale']**2/
                                                                self.parameters['adimensionalization']['time_scale'],self.cons_idx)
    
    def mfp_nn(self,r,z):
        """
        returns value of neutral mean free path in given point (r,z)
        """
        if self.atomic_parameters is None:
                raise ValueError("Please, provide atomic settings for the simulation")
        if "cx" not in self.atomic_parameters.keys():
                raise ValueError("Please, provide charge exchange atomic settings for the simulation")
        if "iz" not in self.atomic_parameters.keys():
                raise ValueError("Please, provide ionization atomic settings for the simulation")
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        solution = np.zeros([1,self.neq])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] ==0:
            return 0
        return calculate_mfp_cons(solution,self.dnn_parameters,self.atomic_parameters,
                                                self._e,self.parameters['adimensionalization']['mass_scale'],
                                                self.parameters['adimensionalization']['temperature_scale'],
                                                self.parameters['adimensionalization']['density_scale'],
                                                self.parameters['physics']['Mref'],
                                                self.parameters['adimensionalization']['length_scale'],
                                                self.parameters['adimensionalization']['time_scale'])

    def p_dyn(self,r,z):
        """
        returns value of dynamic pressure kb(Ti+Te)+mD*u**2 in given point (r,z)
        """

        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        solution = np.zeros([1,self.neq])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] ==0:
            return 0

        return calculate_pdyn_cons(solution,(2/3/self.parameters['physics']['Mref'])* \
                                          self.parameters['adimensionalization']['density_scale']* \
                                          self.parameters['adimensionalization']['temperature_scale']* \
                                          self.parameters['adimensionalization']['charge_scale'],
                                          self.parameters['adimensionalization']['speed_scale']**2* \
                                          self.parameters['adimensionalization']['mass_scale']* \
                                          self.parameters['adimensionalization']['density_scale'],
                                          self.cons_idx)

    def grad_ti(self,r,z,coordinate):
        """
        returns value of derivative of ion temperature over chosen direction in given point (r,z)
        """
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        
        #gradTi = 2/3/Mref*(Q1*(U2**2/U1**3-U3/U1**2)+Q2*(-U2/U1**2)+Q3*(1/U1))
        if coordinate == 'x':
            idx = 0
        elif coordinate == 'y':
            idx = 1
        else:
            raise ValueError(f'{coordinate} is not a coordinate of the problem')
        solution = np.zeros([1,self.neq])
        gradient = np.zeros([1,self.neq,2])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
            for k in range(2):
                gradient[0,i,k] = self._gradient_interpolators[i][k](r,z)
        if solution[0,0] ==0:
            return 0

        return calculate_grad_Ti_cons(solution,gradient,self.parameters['adimensionalization']['temperature_scale'],
                                      self.parameters['physics']['Mref'],self.parameters['adimensionalization']['length_scale'],self._cons_idx)[0][idx]

    def grad_ti_par(self,r,z):
        """
        returns value of derivative of ion temperature over parallel direction (r) in given point (r,z)
        """
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        
        #dTi/dl = gradTi*b
        solution = np.zeros([1,self.neq])
        gradient = np.zeros([1,self.neq,2])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
            for k in range(2):
                gradient[0,i,k] = self._gradient_interpolators[i][k](r,z)
        if solution[0,0] ==0:
            return 0
        Br = self.field_interpolators[0](r,z)
        Bz = self.field_interpolators[1](r,z)
        Bt = self.field_interpolators[2](r,z)

        return calculate_grad_Ti_par_cons(solution,gradient,Br,Bz,Bt,self.parameters['adimensionalization']['temperature_scale'],
              self.parameters['physics']['Mref'],self.parameters['adimensionalization']['length_scale'],self._cons_idx)

    def grad_te(self,r,z,coordinate):
        """
        returns value of derivative of electron temperature over x direction (r) in given point (r,z)
        """
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        
        #gradTi = 2/3/Mref*(Q1*(-U4/U1**2)+Q4*(1/U1))
        if coordinate == 'x':
            idx = 0
        elif coordinate == 'y':
            idx = 1
        else:
            raise ValueError(f'{coordinate} is not a coordinate of the problem')

        solution = np.zeros([1,self.neq])
        gradient = np.zeros([1,self.neq,2])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
            for k in range(2):
                gradient[0,i,k] = self._gradient_interpolators[i][k](r,z)
        if solution[0,0] ==0:
            return 0

        return calculate_grad_Te_cons(solution,gradient,self.parameters['adimensionalization']['temperature_scale'],
                                      self.parameters['physics']['Mref'],self.parameters['adimensionalization']['length_scale'],self._cons_idx)[0][idx]

    def grad_te_par(self,r,z):
        """
        returns value of derivative of electron temperature over parallel direction (r) in given point (r,z)
        """
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        
        #dTi/dl = gradTi*b
        solution = np.zeros([1,self.neq])
        gradient = np.zeros([1,self.neq,2])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
            for k in range(2):
                gradient[0,i,k] = self._gradient_interpolators[i][k](r,z)
        if solution[0,0] ==0:
            return 0
        Br = self._field_interpolators[0](r,z)
        Bz = self._field_interpolators[1](r,z)
        Bt = self._field_interpolators[2](r,z)

        
        return calculate_grad_Te_par_cons(solution,gradient,Br,Bz,Bt,self.parameters['adimensionalization']['temperature_scale'],
              self.parameters['physics']['Mref'],self.parameters['adimensionalization']['length_scale'],self._cons_idx)

    def particle_flux_par(self,r,z):
        """
        returns value of parallel particle flux in given point (r,z)
        """
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()

        # Gamma = n0*u0*U2
        # u = u0*U2/U1
        #only fill needed field
        solution = np.zeros([1,self.neq])
        solution[:,self._cons_idx[b'Gamma']] = self._solution_interpolators[self._cons_idx[b'Gamma']](r,z)
        return calculate_parallel_flux_cons(solution,self.parameters['adimensionalization']['density_scale']* \
                                                     self.parameters['adimensionalization']['speed_scale'],self._cons_idx)
    
    def ion_heat_flux_par_conv(self,r,z):
        """
        returns value of parallel convective ion heat flux in given point (r,z)
        """
        # q_ipar = (5/2*kb*n*Ti+1/2*mD*n*u**2)u

        solution = np.zeros([1,self.neq])

        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] ==0:
            return 0

        return calculate_parallel_ion_heat_flux_par_conv_cons(solution,self.parameters['adimensionalization']['density_scale'],
                                                                  self.parameters['adimensionalization']['temperature_scale'],
                                                                  self.parameters['physics']['Mref'],
                                                                  self.parameters['adimensionalization']['charge_scale'],
                                                                  self.parameters['adimensionalization']['mass_scale'],
                                                                  self.parameters['adimensionalization']['speed_scale'],
                                                                  self._cons_idx)


    def ion_heat_flux_par_cond(self,r,z):
        """
        returns value of parallel conductive ion heat flux in given point (r,z)
        """
        # q_ipar = - kappa_par_i*Ti**(5/2)*dTi/dl

        solution = np.zeros([1,self.neq])
        gradient = np.zeros([1,self.neq,2])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
            for k in range(2):
                gradient[0,i,k] = self._gradient_interpolators[i][k](r,z)
        if solution[0,0] ==0:
            return 0
        Br = self._field_interpolators[0](r,z)
        Bz = self._field_interpolators[1](r,z)
        Bt = self._field_interpolators[2](r,z)
        return calculate_parallel_ion_heat_flux_par_cond_cons(solution,gradient,Br,Bz,Bt,self.parameters['physics']['diff_pari']/(self.parameters['adimensionalization']['time_scale']**3* \
                        self.parameters['adimensionalization']['temperature_scale']**(7/2)/(self.parameters['adimensionalization']['density_scale']*
                        self.parameters['adimensionalization']['length_scale']**4)/self.parameters['adimensionalization']['mass_scale']),
                        self.parameters['adimensionalization']['temperature_scale'],self.parameters['physics']['Mref'],self.parameters['adimensionalization']['length_scale'],
                        50,self._cons_idx)


    
    def ion_heat_flux_par(self,r,z):
        """
        returns value of parallel ion heat flux in given point (r,z)
        """
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()

        # q_ipar = (5/2*kb*n*Ti+1/2*mD*n*u**2)u - kappa_par_i*Ti**(5/2)*dTi/dl


        solution = np.zeros([1,self.neq])
        gradient = np.zeros([1,self.neq,2])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
            for k in range(2):
                gradient[0,i,k] = self._gradient_interpolators[i][k](r,z)
        if solution[0,0] ==0:
            return 0

        Br = self._field_interpolators[0](r,z)
        Bz = self._field_interpolators[1](r,z)
        Bt = self._field_interpolators[2](r,z)

        return calculate_parallel_ion_heat_flux_par_cons(solution,gradient,Br,Bz,Bt,self.parameters['adimensionalization']['density_scale'],self.parameters['physics']['diff_pari']/(self.parameters['adimensionalization']['time_scale']**3* \
                        self.parameters['adimensionalization']['temperature_scale']**(7/2)/(self.parameters['adimensionalization']['density_scale']*
                        self.parameters['adimensionalization']['length_scale']**4)/self.parameters['adimensionalization']['mass_scale']),
                        self.parameters['adimensionalization']['temperature_scale'],self.parameters['physics']['Mref'],
                        self.parameters['adimensionalization']['charge_scale'],self.parameters['adimensionalization']['mass_scale'],
                        self.parameters['adimensionalization']['speed_scale'],self.parameters['adimensionalization']['length_scale'],
                        50,self._cons_idx)

    def electron_heat_flux_par_conv(self,r,z):
        """
        returns value of parallel convective electron heat flux in given point (r,z)
        """
        # q_epar = (5/2*kb*n*Te)u


        solution = np.zeros([1,self.neq])

        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] ==0:
            return 0

        return calculate_parallel_electron_heat_flux_par_conv_cons(solution,self.parameters['adimensionalization']['density_scale'],
                                                                  self.parameters['adimensionalization']['temperature_scale'],
                                                                  self.parameters['physics']['Mref'],
                                                                  self.parameters['adimensionalization']['charge_scale'],
                                                                  self.parameters['adimensionalization']['speed_scale'],
                                                                  self._cons_idx)

    def electron_heat_flux_par_cond(self,r,z):
        """
        returns value of parallel conductive electron heat flux in given point (r,z)
        """
        # q_epar = - kappa_par_e*Te**(5/2)*dTi/dl

        solution = np.zeros([1,self.neq])
        gradient = np.zeros([1,self.neq,2])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
            for k in range(2):
                gradient[0,i,k] = self._gradient_interpolators[i][k](r,z)
        if solution[0,0] ==0:
            return 0
        Br = self._field_interpolators[0](r,z)
        Bz = self._field_interpolators[1](r,z)
        Bt = self._field_interpolators[2](r,z)
        return calculate_parallel_electron_heat_flux_par_cond_cons(solution,gradient,Br,Bz,Bt,self.parameters['physics']['diff_pare']/(self.parameters['adimensionalization']['time_scale']**3* \
                        self.parameters['adimensionalization']['temperature_scale']**(7/2)/(self.parameters['adimensionalization']['density_scale']*
                        self.parameters['adimensionalization']['length_scale']**4)/self.parameters['adimensionalization']['mass_scale']),
                        self.parameters['adimensionalization']['temperature_scale'],self.parameters['physics']['Mref'],self.parameters['adimensionalization']['length_scale'],
                        50,self._cons_idx) 


    def electron_heat_flux_par(self,r,z):
        """
        returns value of parallel electron heat flux in given point (r,z)
        """
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()

        # q_epar = (5/2*kb*n*Te) - kappa_par_e*Te**(5/2)*dTe/dl



        solution = np.zeros([1,self.neq])
        gradient = np.zeros([1,self.neq,2])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
            for k in range(2):
                gradient[0,i,k] = self._gradient_interpolators[i][k](r,z)
        if solution[0,0] ==0:
            return 0

        Br = self._field_interpolators[0](r,z)
        Bz = self._field_interpolators[1](r,z)
        Bt = self._field_interpolators[2](r,z)

        return calculate_parallel_electron_heat_flux_par_cons(solution,gradient,Br,Bz,Bt,self.parameters['adimensionalization']['density_scale'],self.parameters['physics']['diff_pare']/(self.parameters['adimensionalization']['time_scale']**3* \
                        self.parameters['adimensionalization']['temperature_scale']**(7/2)/(self.parameters['adimensionalization']['density_scale']*
                        self.parameters['adimensionalization']['length_scale']**4)/self.parameters['adimensionalization']['mass_scale']),
                        self.parameters['adimensionalization']['temperature_scale'],self.parameters['physics']['Mref'],
                        self.parameters['adimensionalization']['charge_scale'],
                        self.parameters['adimensionalization']['speed_scale'],self.parameters['adimensionalization']['length_scale'],
                        50,self._cons_idx)

        
    
        

    
