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

        self._combined_to_full = False
        self._combined_simple_solution = False
        self._full_phys_initialized = False
        self._simple_phys_initialized = False

        self._solution_glob = None
        self._gradient_glob = None
        self._magnetic_field_glob = None
        self._jtor_glob = None

        self._solution_simple = None
        self._gradient_simple = None
        self._magnetic_field_simple= None
        if 'charge_scale' in self.parameters['adimensionalization'].keys():
            self._e = self.parameters['adimensionalization']['charge_scale']
        else: 
            self._e = 1.60217662e-19
            self.parameters['adimensionalization']['charge_scale'] = self.e

        # defining the indexes of conservative variables
        self._cons_idx = {}
        for i,label in enumerate(self.parameters['physics']['conservative_variable_names']):
            self._cons_idx[label] = i
        self._phys_idx = {}
        for i,label in enumerate(self.parameters['physics']['physical_variable_names']):
            self._phys_idx[label] = i
        
        self._solution_simple_phys = None
        self._gradient_simple_phys = None

        self._solution_glob_phys = None
        self._gradient_glob_phys = None
        self._atomic_parameters = None
        self._dnn_parameters = None


        self._ionization_source_simple = None
        self._ionization_rate_simple = None
        self._ohmic_source_simple = None
        self._cx_rate_simple = None
        self._dnn_simple = None
        self._mfp_simple = None

        self._ionization_source = None
        self._ohmic_source = None
        self._ionization_rate = None
        self._cx_rate = None
        self._dnn = None
        self._mfp = None

        
        self._sample_interpolator = None

        self._solution_interpolators= None
        self._gradient_interpolators= None

        self._solution_gauss = None
        self._gradient_gauss = None
        self._magnetic_field_gauss = None
        self._jtor_gauss = None
        self._ohmic_source_gauss = None
        self._ionization_source_gauss = None
        

        
        
     


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
    def combined_to_full(self):
        """Flag which tells if the solution has been combined to full"""
        return self._combined_to_full

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
    def cx_rate_simple(self):
        """Charge exchange rate coefficient on a simple solution mesh"""
        return self._cx_rate_simple

    @property
    def dnn_simple(self):
        """Neutral diffusion on a simple solution mesh"""
        return self._dnn_simple
    
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
    



    def recombine_full_solution(self):
        """ 
        Recombine raw solutions into one single mesh
        """
        if not self.mesh.combined_to_full:
            print('Comibining first mesh full')
            self.mesh.recombine_full_mesh()

        self._solution_glob = \
            np.zeros((self.mesh._nelems_glob+1,self.mesh.mesh_parameters['nodes_per_element'],self.neq))
        self._gradient_glob = \
            np.zeros((self.mesh._nelems_glob+1,self.mesh.mesh_parameters['nodes_per_element'],self.neq,self.ndim))
        self._magnetic_field_glob = \
            np.zeros((self.mesh._nelems_glob+1,self.mesh.mesh_parameters['nodes_per_element'],3))
        
        self._jtor_glob = \
            np.zeros((self.mesh._nelems_glob+1,self.mesh.mesh_parameters['nodes_per_element']))

        for i in range(self.n_partitions):
            # reshape to the shape of elements
            raw_solution = self.raw_solutions[i].reshape(self.raw_solutions[i].shape[0]//self.mesh.mesh_parameters['nodes_per_element'],self.mesh.mesh_parameters['nodes_per_element'],self.neq)
            # removing ghost elements
            raw_solution = raw_solution[~self.mesh.raw_ghost_elements[i].astype(bool).flatten(),:]
            self._solution_glob[self.mesh.raw_rest_mesh_data[i]['loc2glob_el'][~self.mesh.raw_ghost_elements[i].flatten()]] = raw_solution

            # reshape to the shape of the elements
            raw_gradient = self.raw_gradients[i].reshape(self.raw_gradients[i].shape[0]//self.mesh.mesh_parameters['nodes_per_element'],self.mesh.mesh_parameters['nodes_per_element'],self.neq,self.ndim)
            raw_gradient = raw_gradient[~self.mesh.raw_ghost_elements[i].astype(bool).flatten(),:,:]
            self._gradient_glob[self.mesh.raw_rest_mesh_data[i]['loc2glob_el'][~self.mesh.raw_ghost_elements[i].flatten()]] = raw_gradient

            #magnetic field
            raw_field = self.raw_equilibriums[i]['magnetic_field'][self.mesh.raw_connectivity[i]]
            raw_field = raw_field[~self.mesh.raw_ghost_elements[i].astype(bool).flatten(),:]
            self._magnetic_field_glob[self.mesh.raw_rest_mesh_data[i]['loc2glob_el'][~self.mesh.raw_ghost_elements[i].flatten()]]  = raw_field

            #plasma current
            raw_jtor = self.raw_equilibriums[i]['plasma_current'][self.mesh.raw_connectivity[i]]
            raw_jtor = raw_jtor[~self.mesh.raw_ghost_elements[i].astype(bool).flatten(),:]
            self._jtor_glob[self.mesh.raw_rest_mesh_data[i]['loc2glob_el'][~self.mesh.raw_ghost_elements[i].flatten()]]  = raw_jtor


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
        self._combined_simple_solution = True
    
    def calculate_in_gauss_points(self):
        if not self.combined_to_full:
            print('Comibining first solution full')
            self.recombine_full_solution()
        if self.mesh.reference_element is None:
            raise ValueError("Please, provide reference element to the mesh")

        self._solution_gauss = np.einsum('ij,kjh->kih', self.mesh.reference_element['N'],self.solution_glob)
        self._gradient_gauss = np.einsum('ij,kjhl->kihl', self.mesh.reference_element['N'],self.gradient_glob)
        self._magnetic_field_gauss = np.einsum('ij,kjh->kih', self.mesh.reference_element['N'],self.magnetic_field_glob)
        self._jtor_gauss = np.einsum('ij,kj->ki', self.mesh.reference_element['N'],self.jtor_glob)
        



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
            if (cons_variable != b'Gamma') :
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
                data_loc = data
            rho_conserv = data_loc[:,self.cons_idx[b'rho']]
            gamma_conserv = data_loc[:,self.cons_idx[b'Gamma']]
            nEi_conserv = data_loc[:,self.cons_idx[b'nEi']]
            nEe_conserv = data_loc[:,self.cons_idx[b'nEe']]
            rhon_conserv = data_loc[:,self.cons_idx[b'rhon']]
            for i in range(self.nphys):
                phys_variable =self.parameters['physics']['physical_variable_names'][i]
                if (phys_variable == b'rho'):
                    # n = n_0*U1 (indexing for U in this comments as in fortran)
                    solution_phys[:,i] = rho_conserv*self.parameters['adimensionalization']['density_scale']
                elif (phys_variable == b'u'):
                    # u = u_0*U2/U1
                    solution_phys[:,i] = gamma_conserv/rho_conserv*self.parameters['adimensionalization']['speed_scale']
                elif (phys_variable == b'Ei'):
                    # Ei = m_0*u_0**2*U3/U1
                    solution_phys[:,i] = nEi_conserv/rho_conserv*self.parameters['adimensionalization']['speed_scale']**2*self.parameters['adimensionalization']['mass_scale']
                elif (phys_variable == b'Ee'):
                    # Ee = m_0*u_0**2*U4/U1
                    solution_phys[:,i] = nEe_conserv/rho_conserv*self.parameters['adimensionalization']['speed_scale']**2*self.parameters['adimensionalization']['mass_scale']
                elif (phys_variable == b'pi'):
                    # pi = 2/3/Mref*e*T0*(U3-1/2*U2**2/U1)
                    solution_phys[:,i] = nEi_conserv - 0.5*gamma_conserv**2/rho_conserv
                    solution_phys[:,i] *= (2/3/self.parameters['physics']['Mref'])
                    solution_phys[:,i] *= self.parameters['adimensionalization']['density_scale']* \
                                          self.parameters['adimensionalization']['temperature_scale']*self.parameters['adimensionalization']['charge_scale']
                elif (phys_variable == b'pe'):
                    # pe = 2/3/Mref*e*T0*U4
                    solution_phys[:,i] = nEe_conserv
                    solution_phys[:,i] *= (2/3/self.parameters['physics']['Mref'])
                    solution_phys[:,i] *= self.parameters['adimensionalization']['density_scale']* \
                                                       self.parameters['adimensionalization']['temperature_scale']*self.parameters['adimensionalization']['charge_scale']
                elif (phys_variable == b'Ti'):
                    # Ti = 2/3/Mref*e*T0*(U3-1/2*U2**2/U1)/U1
                    solution_phys[:,i] = (nEi_conserv - 0.5*gamma_conserv**2/rho_conserv)/rho_conserv
                    solution_phys[:,i] *= (2/3/self.parameters['physics']['Mref'])
                    solution_phys[:,i] *= self.parameters['adimensionalization']['temperature_scale']
                elif (phys_variable == b'Te'):
                    # Te = 2/3/Mref*e*T0*U4/U1
                    solution_phys[:,i] = nEe_conserv/rho_conserv
                    solution_phys[:,i] *= (2/3/self.parameters['physics']['Mref'])
                    solution_phys[:,i] *= self.parameters['adimensionalization']['temperature_scale']
                elif (phys_variable == b'Csi'):
                    # cs = u0*(2/3*(U3+U4-1/2*U2**2/U1)/U1)**0.5
                    ti = (nEi_conserv - 0.5*gamma_conserv**2/rho_conserv)/rho_conserv*(2/3)
                    te = nEe_conserv/rho_conserv*(2/3)
                    solution_phys[:,i] = np.sqrt((ti+te))
                    solution_phys[:,i] *= self.parameters['adimensionalization']['speed_scale']
                elif (phys_variable == b'M'):
                    # M = u/cs
                    ti = (nEi_conserv - 0.5*gamma_conserv**2/rho_conserv)/rho_conserv*(2/3)
                    te = nEe_conserv/rho_conserv*(2/3)
                    u = gamma_conserv/rho_conserv
                    cs = np.sqrt((ti+te))
                    solution_phys[:,i] = u/cs
                elif (phys_variable == b'rhon'):
                    # n_n = n_0*U5
                    solution_phys[:,i] = rhon_conserv*self.parameters['adimensionalization']['density_scale']
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
                data_loc = data
                sol_loc = self.solution_simple
            rho_conserv = sol_loc[:,self.cons_idx[b'rho']][None].T
            rho_conserv_grad = data_loc[:,self.cons_idx[b'rho'],:]
            gamma_conserv = sol_loc[:,self.cons_idx[b'Gamma']][None].T
            gamma_conserv_grad = data_loc[:,self.cons_idx[b'Gamma'],:]
            nEi_conserv = sol_loc[:,self.cons_idx[b'nEi']][None].T
            nEi_conserv_grad = data_loc[:,self.cons_idx[b'nEi'],:]
            nEe_conserv = sol_loc[:,self.cons_idx[b'nEe']][None].T
            nEe_conserv_grad = data_loc[:,self.cons_idx[b'nEe'],:]
            rhon_conserv = sol_loc[:,self.cons_idx[b'rhon']][None].T
            rhon_conserv_grad = data_loc[:,self.cons_idx[b'rhon'],:]
            for i in range(self.nphys):
                phys_variable =self.parameters['physics']['physical_variable_names'][i]
                if (phys_variable == b'rho'):
                    # grad(n) = n0/L0*grad(U1)
                    grad_phys[:,i,:] = rho_conserv_grad*self.parameters['adimensionalization']['density_scale']
                elif (phys_variable == b'u'):
                    # grad(u) = u0/L0*(-1*grad(U1)*U2/U1**2+grad(U2)/(U1))
                    grad_phys[:,i,:] = rho_conserv_grad*(-1.*gamma_conserv/rho_conserv**2)+gamma_conserv_grad*(1/rho_conserv)
                    grad_phys[:,i,:] *= self.parameters['adimensionalization']['speed_scale']
                elif (phys_variable == b'Ei'):
                    # grad(Ei) = m_i*u0**2/L0*(-1*grad(U1)*U3/U1**2+grad(U3)/U1)
                    grad_phys[:,i,:] = rho_conserv_grad*(-1.*nEi_conserv/rho_conserv**2)+nEi_conserv_grad*(1/rho_conserv)
                    grad_phys[:,i,:] *= self.parameters['adimensionalization']['speed_scale']**2*self.parameters['adimensionalization']['mass_scale']
                elif (phys_variable == b'Ee'):
                    # grad(Ee) = m_i*u0**2/L0*(-1*grad(U1)*U4/U1**2+grad(U4)/U1)
                    grad_phys[:,i,:] = rho_conserv_grad*(-1.*nEe_conserv/rho_conserv**2)+nEe_conserv_grad*(1/rho_conserv)
                    grad_phys[:,i,:] *= self.parameters['adimensionalization']['speed_scale']**2*self.parameters['adimensionalization']['mass_scale']
                elif (phys_variable == b'pi'):
                    # grad(pi) = 2/3/Mref*e*T0/L0*(grad(U3)-grad(U2)*U2/U1+1/2*grad(U1)*U2**2/U1**2)
                    grad_phys[:,i,:] = nEi_conserv_grad - gamma_conserv_grad*gamma_conserv**2/rho_conserv+\
                                        0.5*rho_conserv_grad*gamma_conserv**2/rho_conserv**2
                    grad_phys[:,i,:] *= (2/3/self.parameters['physics']['Mref'])
                    grad_phys[:,i,:] *= self.parameters['adimensionalization']['density_scale']* \
                                          self.parameters['adimensionalization']['temperature_scale']*self.parameters['adimensionalization']['charge_scale']
                elif (phys_variable == b'pe'):
                    # grad(pe) = 2/3/Mref*e*T0/L0*(grad(U4))
                    grad_phys[:,i,:] = nEe_conserv_grad
                    grad_phys[:,i,:] *= (2/3/self.parameters['physics']['Mref'])
                    grad_phys[:,i,:] *= self.parameters['adimensionalization']['density_scale']* \
                                                       self.parameters['adimensionalization']['temperature_scale']*self.parameters['adimensionalization']['charge_scale']
                elif (phys_variable == b'Ti'):
                    # assuming that Ei and u already calculated
                    # grad(Ti) = T0/L0*2/3/Mref*(grad(Ei)/m_i/u0**2-grad(u)*u/u0**2)
                    Ei_grad = grad_phys[:,self.phys_idx[b'Ei'],:]/self.parameters['adimensionalization']['speed_scale']**2/self.parameters['adimensionalization']['mass_scale']
                    u_grad = grad_phys[:,self.phys_idx[b'u'],:]/self.parameters['adimensionalization']['speed_scale']
                    u = gamma_conserv/rho_conserv
                    grad_phys[:,i,:] = Ei_grad-u_grad*u
                    grad_phys[:,i,:] *= (2/3/self.parameters['physics']['Mref'])
                    grad_phys[:,i,:] *= self.parameters['adimensionalization']['temperature_scale']
                elif (phys_variable == b'Te'):
                    # assuming that Ee already calculated
                    # grad(Te) = T0/L0*2/3/Mref*(grad(Ee)/m_i/u0**2)
                    Ee_grad = grad_phys[:,self.phys_idx[b'Ee'],:]/self.parameters['adimensionalization']['speed_scale']**2/self.parameters['adimensionalization']['mass_scale']
                    grad_phys[:,i,:] = Ee_grad
                    grad_phys[:,i,:] *= (2/3/self.parameters['physics']['Mref'])
                    grad_phys[:,i,:] *= self.parameters['adimensionalization']['temperature_scale']
                elif (phys_variable == b'Csi'):
                    # assuming cs already calculated
                    #cs = u0*(2/3*(U3+U4-1/2*U2**2/U1)/U1)**0.5
                    # grad(cs) = u0/L0/2/(cs/u0)*(2/3)*(grad(U1)*(-U3/U1**2-U4/U1**2+U2**2/U1**3)+
                    #                                   grad(U2)*(-U2/U1**2)+grad(U3)/U1+grad(U4)/U1)
                    ti = (nEi_conserv - 0.5*gamma_conserv**2/rho_conserv)/rho_conserv*(2/3)
                    te = nEe_conserv/rho_conserv*(2/3)
                    cs = np.sqrt((ti+te))
                    grad_phys[:,i,:] = 1/2/cs*(2/3)*(rho_conserv_grad*((-1*nEi_conserv-nEe_conserv+gamma_conserv**2/rho_conserv)/rho_conserv**2)+
                                               gamma_conserv_grad*(-1*gamma_conserv/rho_conserv**2)+nEi_conserv_grad/rho_conserv+nEe_conserv_grad/rho_conserv)
                    grad_phys[:,i,:] *= self.parameters['adimensionalization']['speed_scale']
                elif (phys_variable == b'M'):
                    # assuming cs and u already calculated
                    # M = u/cs
                    # grad(M) = grad(u)/cs-grad(cs)*u/cs**2
                    u = gamma_conserv/rho_conserv
                    grad_u = grad_phys[:,self.phys_idx[b'u'],:]
                    ti = (nEi_conserv - 0.5*gamma_conserv**2/rho_conserv)/rho_conserv*(2/3)
                    te = nEe_conserv/rho_conserv*(2/3)
                    cs = np.sqrt((ti+te))
                    grad_cs = grad_phys[:,self.phys_idx[b'Csi'],:]
                    grad_phys[:,i,:] = grad_u/cs+grad_cs*u/cs**2
                elif (phys_variable == b'rhon'):
                    #grad(n_n) = n0/L0*grad(U5)
                    rhon_conserv_grad = data_loc[:,self.cons_idx[b'rhon'],:]
                    grad_phys[:,i,:] = rhon_conserv_grad
                    grad_phys[:,i,:] *= self.parameters['adimensionalization']['density_scale']
                    

            grad_phys/=self.parameters['adimensionalization']['length_scale']

            if len(data.shape) == 4:
                    self._gradient_glob_phys = grad_phys.reshape((data.shape[0],data.shape[1],self.nphys,self.ndim))
            elif len(data.shape) == 3:
                self._gradient_simple_phys = grad_phys
                                                    
    def plot_overview_physical(self,n_levels=100):
            """
            Plot n, n_n, Ti, Te, M
            As a physical overview legacy
            """

            if not self._simple_phys_initialized:
                print('Initializing physical solution first')
                self.init_phys_variables('simple')



            colorbar_labels = [r'n, m$^{-3}$',r'$n_n$, m$^{-3}$',r'$T_i$',r'$T_e$',r'M']
            solutions_plot = np.zeros_like(self.solution_simple)
            solutions_plot[:,0] = self.solution_simple_phys[:,0] #ne
            solutions_plot[:,1] = self.solution_simple_phys[:,-1] #n_n
            solutions_plot[:,2] = self.solution_simple_phys[:,6] #Ti
            solutions_plot[:,3] = self.solution_simple_phys[:,7] #Te
            solutions_plot[:,4] = self.solution_simple_phys[:,9] #M




            #additional triangulaton: since we have more than 3 points in each element, we can triangulate it
            #take any triangle from the mesh
            if self.mesh.connectivity_big is None:
                self.mesh.create_connectivity_big()


            n_lines = int(np.floor(self.neq/2+0.5))
            fig, axes = plt.subplots(n_lines,2, figsize = (15,7.5*n_lines))

            for i in range(self.neq):
                if ((i!=3)and(i!=4)and(i!=2)) :
                    axes[i//2,i%2] = self.mesh.plot_full_mesh(solutions_plot[:,i],ax=axes[i//2,i%2],
                                                              log=True,label=colorbar_labels[i],connectivity=self.mesh.connectivity_big,n_levels=n_levels)
                else:
                    axes[i//2,i%2] = self.mesh.plot_full_mesh(solutions_plot[:,i],ax=axes[i//2,i%2],
                                                              log=False,label=colorbar_labels[i],connectivity=self.mesh.connectivity_big,n_levels=n_levels)
    

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


            colorbar_labels = [r'n, m$^{-3}$',r'$n_n$, m$^{-3}$',r'$T_i$',r'$T_e$',r'M']
            solutions_plot = np.zeros_like(self.solution_simple)
            solutions_plot[:,0] = self.solution_simple_phys[:,0]-second_solution.solution_simple_phys[:,0] #ne
            solutions_plot[:,1] = self.solution_simple_phys[:,-1]-second_solution.solution_simple_phys[:,-1] #n_n
            solutions_plot[:,2] = self.solution_simple_phys[:,6]-second_solution.solution_simple_phys[:,6] #Ti
            solutions_plot[:,3] = self.solution_simple_phys[:,7]-second_solution.solution_simple_phys[:,7] #Te
            solutions_plot[:,4] = self.solution_simple_phys[:,9]-second_solution.solution_simple_phys[:,9] #M




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
    
    def calculate_variables_along_line(self,r_line,z_line,variable_list):
        """
        calculates plasma parameters noted in variables list on a given line
        returns a dictionary with variables as keys and values along lines for them
        """
        defined_variables = ['n','nn','te','ti','M','dnn','mfp','u',
                             'p_dyn','q_i_par','q_e_par','gamma',
                             'q_i_par_conv','q_i_par_cond',
                             'q_e_par_conv','q_e_par_cond']
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
            else:
                raise KeyError(f'{variable} is not in the list of posible variables:  {defined_variables}')
            result[variable] = temp
        return result



    def save_summary_line(self,save_folder,r_line,z_line,variable_list):


        defined_variables = ['n','nn','te','ti','M','dnn','mfp','u',
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
            else:
                raise KeyError(f'{variable} is not in the list of posible variables')

        return values_on_line

    



    def calculate_ohmic_source(self, which='simple'):
        """
            calculate the ionization rate
        """

        if 'ohmic_coeff' not in  self.parameters['physics'].keys():
            raise KeyError('Please, provide ohmic heating adimensionalized coefficient')
        
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
                                                            self.parameters['physics']['ohmic_coeff'])
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
                                                            self.parameters['physics']['ohmic_coeff'])
            
            


        



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
        if self.mesh._element_number_mask is None:
            print('Defining an element number mask')
            el_numbers = np.repeat(np.arange(len(self.mesh.connectivity_glob)),self.mesh.connectivity_big.shape[0]/self.mesh.connectivity_glob.shape[0])
        
            self.mesh._element_number_mask = Discrete2DMesh(self.mesh.vertices_glob, self.mesh.connectivity_big,
                      el_numbers,limit=False,default_value = -1)
        if self._sample_interpolator is None:
            self._sample_interpolator = SoledgeHDG2DInterpolator(self.mesh.vertices_glob,np.ones_like(self.solution_glob[:,:,0]),self.mesh.connectivity_glob,
                self.mesh.element_number_mask,self.mesh.reference_element['NodesCoord'],self.mesh.mesh_parameters['element_type'], self.mesh.p_order,limit=False)
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

    def n(self,r,z):
        """
        returns value of density in given point (r,z)
        """
        if b'rho' not in self.parameters['physics']['physical_variable_names']:
            raise KeyError('density is not in the models')
        
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        # n0*U1
        return self.parameters['adimensionalization']['density_scale']*self._solution_interpolators[self._cons_idx[b'rho']](r,z)

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
        u1 = self._solution_interpolators[self._cons_idx[b'rho']](r,z)
        if u1 == 0:
            return 0
        u2 = self._solution_interpolators[self._cons_idx[b'Gamma']](r,z)
        u3 = self._solution_interpolators[self._cons_idx[b'nEi']](r,z)
        return self.parameters['adimensionalization']['temperature_scale']*2/3/self.parameters['physics']['Mref']*(u3/u1-1/2*u2**2/u1**2)

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
        u1 = self._solution_interpolators[self._cons_idx[b'rho']](r,z)
        if u1 == 0:
            return 0

        u4 = self._solution_interpolators[self._cons_idx[b'nEe']](r,z)
        return self.parameters['adimensionalization']['temperature_scale']*2/3/self.parameters['physics']['Mref']*(u4/u1)
    
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
        u1 = self._solution_interpolators[self._cons_idx[b'rho']](r,z)
        if u1 == 0:
            return 0
        u2 = self._solution_interpolators[self._cons_idx[b'Gamma']](r,z)
        return self.parameters['adimensionalization']['speed_scale']*(u2/u1)
    
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
        u1 = self._solution_interpolators[self._cons_idx[b'rho']](r,z)
        if u1 == 0:
            return 0
        u2 = self._solution_interpolators[self._cons_idx[b'Gamma']](r,z)
        u3 = self._solution_interpolators[self._cons_idx[b'nEi']](r,z)
        u4 = self._solution_interpolators[self._cons_idx[b'nEe']](r,z)
        return self.parameters['adimensionalization']['speed_scale']*np.sqrt((u3+u4-1/2*u2**2/u1)/u1)
    
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
        cs = self.cs(r,z)
        if cs == 0:
            return 0
        return self.u(r,z)/cs
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

        return self.parameters['adimensionalization']['density_scale']*self._solution_interpolators[self._cons_idx[b'rhon']](r,z)

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
        diff_nn = self.dnn(r,z)
        ti = self.ti(r,z)
        return (2*diff_nn)/np.sqrt(self._e*ti/self.parameters['adimensionalization']['mass_scale'])

    def p_dyn(self,r,z):
        """
        returns value of dynamic pressure kb(Ti+Te)+mD*u**2 in given point (r,z)
        """

        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()

        pressure = self.n(r,z)*self.e*(self.ti(r,z)+self.te(r,z))+self.parameters['adimensionalization']['mass_scale']*self.n(r,z)*self.u(r,z)
        return pressure

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
        q1 = self._gradient_interpolators[0][idx](r,z)
        q2 = self._gradient_interpolators[1][idx](r,z)
        q3 = self._gradient_interpolators[2][idx](r,z)
        u1 = self._solution_interpolators[0](r,z)
        u2 = self._solution_interpolators[1](r,z)
        u3 = self._solution_interpolators[2](r,z)
        #print(r,z)
        #print(u1,u2,u3)

        ti_grad = q1*(u2**2/u1**3-u3/u1**2)+q2*(-1*u2/u1**2)+q3/u1
        ti_grad *= ((2/3/self.parameters['physics']['Mref'])*self.parameters['adimensionalization']['temperature_scale']/
                    self.parameters['adimensionalization']['length_scale'])
        return ti_grad

    def grad_ti_par(self,r,z):
        """
        returns value of derivative of ion temperature over parallel direction (r) in given point (r,z)
        """
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        
        #dTi/dl = gradTi*b
        
        Br = self.field_interpolators[0](r,z)
        Bz = self.field_interpolators[1](r,z)
        Bt = self.field_interpolators[2](r,z)
        br = Br/np.sqrt(Br**2+Bz**2+Bt**2)
        bz = Bz/np.sqrt(Br**2+Bz**2+Bt**2)
       
        return self.grad_ti(r,z,'x')*br+self.grad_ti(r,z,'y')*bz

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
        q1 = self._gradient_interpolators[0][idx](r,z)

        q4 = self._gradient_interpolators[3][idx](r,z)
        u1 = self._solution_interpolators[0](r,z)

        u4 = self._solution_interpolators[3](r,z)

        te_grad = q1*(-1*u4/u1**2)+q4/u1
        te_grad *= ((2/3/self.parameters['physics']['Mref'])*self.parameters['adimensionalization']['temperature_scale']/
                    self.parameters['adimensionalization']['length_scale'])
        return te_grad

    def grad_te_par(self,r,z):
        """
        returns value of derivative of electron temperature over parallel direction (r) in given point (r,z)
        """
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        
        #dTi/dl = gradTi*b
        
        Br = self._field_interpolators[0](r,z)
        Bz = self._field_interpolators[1](r,z)
        Bt = self._field_interpolators[2](r,z)
        br = Br/np.sqrt(Br**2+Bz**2+Bt**2)
        bz = Bz/np.sqrt(Br**2+Bz**2+Bt**2)
       
        return self.grad_te(r,z,'x')*br+self.grad_te(r,z,'y')*bz

    def particle_flux_par(self,r,z):
        """
        returns value of parallel particle flux in given point (r,z)
        """
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()

        # Gamma = n0*u0*U2

        return self.parameters['adimensionalization']['density_scale']* \
               self.parameters['adimensionalization']['speed_scale']*self._solution_interpolators[1](r,z)
    
    def ion_heat_flux_par_conv(self,r,z):
        """
        returns value of parallel convective ion heat flux in given point (r,z)
        """
        # q_ipar = (5/2*kb*n*Ti+1/2*mD*n*u**2)u
        u = self.u(r,z)
        ti = self.ti(r,z)
        
        q_ipar_conv =  self.n(r,z)*(5/2*self.e*ti+ \
                 0.5*self.parameters['adimensionalization']['mass_scale']*u**2)*u
        return q_ipar_conv

    def ion_heat_flux_par_cond(self,r,z):
        """
        returns value of parallel conductive ion heat flux in given point (r,z)
        """
        # q_ipar = - kappa_par_i*Ti**(5/2)*dTi/dl
        ti = min(50,self.ti(r,z))
        
        q_ipar_cond = -1*ti**(5/2)*self.grad_ti_par(r,z)
        q_ipar_cond *=self.parameters['physics']['diff_pari']/(self.parameters['adimensionalization']['time_scale']**3* \
                        self.parameters['adimensionalization']['temperature_scale']**(7/2)/(self.parameters['adimensionalization']['density_scale']*
                        self.parameters['adimensionalization']['length_scale']**4)/self.parameters['adimensionalization']['mass_scale'])
        return q_ipar_cond

    
    def ion_heat_flux_par(self,r,z):
        """
        returns value of parallel ion heat flux in given point (r,z)
        """
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()

        # q_ipar = (5/2*kb*n*Ti+1/2*mD*n*u**2)u - kappa_par_i*Ti**(5/2)*dTi/dl

        #conductive part
        q_ipar = self.ion_heat_flux_par_cond(r,z)

        #convective part
        
        q_ipar += self.ion_heat_flux_par_conv(r,z)

        return q_ipar

    def electron_heat_flux_par_conv(self,r,z):
        """
        returns value of parallel convective electron heat flux in given point (r,z)
        """
        # q_ipar = (5/2*kb*n*Te)u
        u = self.u(r,z)
        te = self.te(r,z)
        
        q_epar_conv =  self.n(r,z)*(5/2*self.e*te)*u
        return q_epar_conv

    def electron_heat_flux_par_cond(self,r,z):
        """
        returns value of parallel conductive electron heat flux in given point (r,z)
        """
        # q_ipar = - kappa_par_i*Ti**(5/2)*dTi/dl
        te = min(50,self.te(r,z))
        
        q_epar_cond = -1*te**(5/2)*self.grad_te_par(r,z)
        q_epar_cond *= self.parameters['physics']['diff_pare']/(self.parameters['adimensionalization']['time_scale']**3* \
                        self.parameters['adimensionalization']['temperature_scale']**(7/2)/(self.parameters['adimensionalization']['density_scale']*
                        self.parameters['adimensionalization']['length_scale']**4)/self.parameters['adimensionalization']['mass_scale'])
        return q_epar_cond

    def electron_heat_flux_par(self,r,z):
        """
        returns value of parallel electron heat flux in given point (r,z)
        """
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()

        # q_epar = (5/2*kb*n*Te) - kappa_par_e*Te**(5/2)*dTe/dl

        #conductive part
        q_epar = self.electron_heat_flux_par_cond(r,z)
        #convective part        
        q_epar += self.electron_heat_flux_par_conv(r,z)

        return q_epar
        
    
        

    
