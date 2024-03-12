import numpy as np
import matplotlib.pyplot as plt
from hdg_postprocess.routines.atomic import *
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
            self._raw_gradients.append(np.swapaxes(raw_gradient.reshape(raw_gradient.shape[0]//(self.ndim*self.neq),self.ndim,self.neq),2,1))

        self._initial_setup()

    def _initial_setup(self):

        self._combined_to_full = False
        self._combined_simple_solution = False
        self._full_phys_initialized = False
        self._simple_phys_initialized = False

        self._solution_glob = None
        self._gradient_glob = None

        self._solution_simple = None
        self._gradient_simple = None
        if 'charge_scale' in self.parameters['adimensionalization'].keys():
            self._e = self.parameters['adimensionalization']['charge_scale']
        else: 
            self._e = 1.60217662e-19
            self.parameters['adimensionalization']['charge_scale'] = self.e
        
        self._solution_simple_phys = None
        self._gradient_simple_phys = None

        self._solution_glob_phys = None
        self._gradient_glob_phys = None
        self._atomic_parameters = None


        self._ionization_source_simple = None
        self._sigma_iz_simple = None

        
        
     


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
    def sigma_iz_simple(self):
        """Ionization rate coefficient on a simple solution mesh"""
        return self._sigma_iz_simple


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

        self._combined_simple_solution = True

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

        # defining the indexes of conservative variables
        self._cons_idx = {}
        for i,label in enumerate(self.parameters['physics']['conservative_variable_names']):
            self._cons_idx[label] = i
        self._phys_idx = {}
        for i,label in enumerate(self.parameters['physics']['physical_variable_names']):
            self._phys_idx[label] = i
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

    def calculate_ionization_source(self,which="simple"):
        """
            calculate the ionization source
            simple: for simple mesh solution
            full: on full mesh solution (to be done)
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
            
            self.calculate_ionization_source(which="full")

            self._ionization_source_simple = np.zeros(self.mesh.vertices_glob.shape[0])
            self._ionization_source_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel(),:] = self._ionization_source.reshape(self.solution_glob.shape[0]*self.solution_glob.shape[1],self.neq)
            
        if which =="full":
            if not self._combined_to_full:
                self.recombine_full_solution()
            self._ionization_source = calculate_iz_source_cons(self.solution_glob,self.atomic_parameters['iz'],
                                                                self.parameters['adimensionalization']['temperature_scale'],
                                                                self.parameters['adimensionalization']['density_scale'],
                                                                self.parameters['physics']['Mref'])
