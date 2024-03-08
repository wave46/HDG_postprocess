import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
class HDGsolution:
    ""
    
    def __init__(self,raw_solutions, raw_solutions_skeleton, raw_gradients,
                 raw_equilibriums,raw_solution_boundary_infos, parameters, 
                 n_partitions, mesh):
        self._parameters = parameters
        self._neq = parameters['Neq'][0]
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

        self._solutions_glob = None
        self._gradients_glob = None

        self._solution_simple = None
        self._gradient_simple = None
        
     


    @property
    def parameters(self):
        """Dictionary with solution parameters"""
        return self._parameters

    @property
    def neq(self):
        """number of equations"""
        return self._neq

    @property
    def ndim(self):
        """number of dimensions"""
        return self._ndim

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
        """raw equilibrium dictionaries on nodes on partitions"""
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
    def combined_to_full(self):
        """Flag which tells if the solution has been combined to full"""
        return self._combined_to_full

    @property
    def combined_simple_solution(self):
        """Flag which tells if the solution has been combined to simple one on full mesh"""
        return self._combined_simple_solution

    @property
    def solutions_glob(self):
        """Solution recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x neq]"""
        return self._solutions_glob

    @property
    def gradients_glob(self):
        """Gradients recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x neq x ndim]"""
        return self._gradients_glob


    def recombine_full_solution(self):
        """ 
        here we combine our raw solutions onto a single mesh
        """
        if not self.mesh.combined_to_full:
            print('Comibining first mesh full')
            self.mesh.recombine_full_mesh()

        self._solutions_glob = \
            np.zeros((self.mesh._nelems_glob+1,self.mesh.mesh_parameters['nodes_per_element'],self.neq))
        self._gradients_glob = \
            np.zeros((self.mesh._nelems_glob+1,self.mesh.mesh_parameters['nodes_per_element'],self.neq,self.ndim))

        for i in range(self.n_partitions):
            # reshape to the shape of elements
            raw_solution = self.raw_solutions[i].reshape(self.raw_solutions[i].shape[0]//self.mesh.mesh_parameters['nodes_per_element'],self.mesh.mesh_parameters['nodes_per_element'],self.neq)
            # removing ghost elements
            raw_solution = raw_solution[~self.mesh.raw_ghost_elements[i].astype(bool).flatten(),:]
            self._solutions_glob[self.mesh.raw_rest_mesh_data[i]['loc2glob_el'][~self.mesh.raw_ghost_elements[i].flatten()]] = raw_solution

            # reshape to the shape of the elements
            raw_gradient = self.raw_gradients[i].reshape(self.raw_gradients[i].shape[0]//self.mesh.mesh_parameters['nodes_per_element'],self.mesh.mesh_parameters['nodes_per_element'],self.neq,self.ndim)
            raw_gradient = raw_gradient[~self.mesh.raw_ghost_elements[i].astype(bool).flatten(),:,:]
            self._gradients_glob[self.mesh.raw_rest_mesh_data[i]['loc2glob_el'][~self.mesh.raw_ghost_elements[i].flatten()]] = raw_gradient


        self._combined_to_full = True

    def recombine_simple_full_solution(self):
        """
        in this routine we reduce the size of solution and gradient
        """
        if not self.combined_to_full:
            print('Comibining first solution full')
            self.recombine_full_solution()
        self._solution_simple = np.zeros([self.mesh.vertices_glob.shape[0],self.neq])
        #back reshaping
        self._solution_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel(),:] = self.solutions_glob.reshape(self.solutions_glob.shape[0]*self.solutions_glob.shape[1],self.neq)

        self._gradient_simple = np.zeros([self.mesh.vertices_glob.shape[0],self.neq,self.ndim])
        self._gradient_simple[self.mesh.connectivity_glob.reshape(-1,1).ravel(),:,:] = self.gradients_glob.reshape(self.solutions_glob.shape[0]*self.solutions_glob.shape[1],self.neq,self.ndim)

        self._combined_simple_solution = True

    def plot_overview(self,n_levels=100):
        """
        here we plot all conservative variables (dimensional) to have a view on our data
        we also leave the solutions adimensional, providing the dimensional ones as outputs
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
                NameError('Unknown conservative varibale')

        #additional triangulaton: since we have more than 3 points in each element, we can triangulate it
        #take any triangle from the mesh
        points = self.mesh.vertices_glob[self.mesh.connectivity_glob[0,:]]
        tri = Delaunay(points)
        connectivity_big = self.mesh.connectivity_glob[:,tri.simplices]
        connectivity_big = connectivity_big.reshape(connectivity_big.shape[0]*connectivity_big.shape[1],3)

        n_lines = int(np.floor(self.neq/2+0.5))
        fig, axes = plt.subplots(n_lines,2, figsize = (15,7.5*n_lines))

        for i in range(self.neq):
            cons_variable =self.parameters['physics']['conservative_variable_names'][i]
            if (cons_variable != b'Gamma') :
                axes[i//2,i%2] = self.mesh.plot_full_mesh(solutions_dimensional[:,i],ax=axes[i//2,i%2],
                                                          log=True,label=colorbar_labels[i],connectivity=connectivity_big)
            else:
                axes[i//2,i%2] = self.mesh.plot_full_mesh(solutions_dimensional[:,i],ax=axes[i//2,i%2],
                                                          log=False,label=colorbar_labels[i],connectivity=connectivity_big)
  

        return fig,axes, solutions_dimensional
        
                
        
        

        


    def cons2phys(self):
        ''' 
        converts conservative adymensional SOLEDGE-HDG solutions into physical adimentionalized ones
        :param conservative: u_conservative [n_points x n_equations] for example {n, nu, nEi, nEe} 
                             for 4 equations
        :param Mref: reference Mach number
        :param neq: number of equations

        :return physical: physical SOLEDGE-HDG solutions
        for example, for n-Gamma-Ti-Te-neutral model
        u_conservative [n_points x n_equations] for example {n, nu, nEi, nEe,n0} for 4 equations
        u_physical [n_ponts x n_phys_variables] for example {n , u, Ei, Ee, pi, pe, Ti, Te, cs, Mach, n0}
        '''

        self._nphys = self.parameters['physics']


