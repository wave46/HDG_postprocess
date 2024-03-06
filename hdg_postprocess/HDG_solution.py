import numpy as np
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
            self._raw_gradients.append(raw_gradient.reshape(raw_gradient.shape[0]//(self.ndim*self.neq),self.ndim,self.neq))

        self._inital_setup()

    def _initial_setup(self):

        self._combined_to_one_mesh = False
     


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
    def solution_simple(self):
        """solution simply united on a single mesh (means not taking into account repeating points"""
        return self._solution_simple

    def combine_to_one_simple(self):
        """ 
        here we combine our raw solutions onto a single mesh
        """
        if self._n_partitions>1:
            self._solution_united = np.empty((0,self.neq))
            for raw_solution, raw_solution_skeleton, raw_gradient, \
                 raw_equilibium,boundary_flag,ghost_elem,ghost_face,\
                 raw_vertices, raw_connectivity, raw_connectivity_boundary, \
                  in \
                 zip(self.raw_solutions,self.raw_solutions_skeleton,self.raw_gradients,
                     self.raw_equilibriums,self.mesh.raw_boundary_flags,self.mesh.raw_ghost_elements,self.mesh.raw_ghost_faces,
                     self.mesh.raw_vertices,self.mesh.raw_connectivity,self.raw_connectivity_boundary):

                solution_simple = np.zeros([raw_vertices.shape[0],raw_solution.shape[1]])
                solution_simple[raw_connectivity.reshape(-1,1).ravel(),:] = raw_solution
                

            


