import numpy as np

from hdg_postprocess.solution_operations import analysis as analysis_ops
from hdg_postprocess.solution_operations import assembly as assembly_ops
from hdg_postprocess.solution_operations import boundary as boundary_ops
from hdg_postprocess.solution_operations import magnetic_equilibrium as equilibrium_ops
from hdg_postprocess.solution_operations import neutrals as neutrals_ops
from hdg_postprocess.solution_operations import physical as physical_ops
from hdg_postprocess.solution_operations import plasma_sources as plasma_source_ops
from hdg_postprocess.solution_operations import plotting as plotting_ops
from hdg_postprocess.solution_operations import pointwise_fields as pointwise_fields_ops
from hdg_postprocess.solution_operations import sampling as sampling_ops
from hdg_postprocess.solution_operations import turbulent_model as turbulent_model_ops
from hdg_postprocess.solution_compat import attach_compat_properties
from hdg_postprocess.view_containers import (
    AtomicRateState,
    InterpolatorState,
    ParameterState,
    SolutionSummaryState,
    SolutionViews,
)


class HDGsolution:
    ""
    def __init__(self,raw_solutions, raw_solutions_skeleton, raw_gradients,
                 raw_equilibriums,raw_solution_boundary_infos, parameters, 
                 n_partitions, mesh):
        self._store_input_metadata(parameters, n_partitions, raw_equilibriums, raw_solution_boundary_infos, mesh)
        self._store_raw_partitions(raw_solutions, raw_solutions_skeleton, raw_gradients)
        self._initial_setup()

    def _store_input_metadata(self, parameters, n_partitions, raw_equilibriums, raw_solution_boundary_infos, mesh):
        self._parameters = parameters
        self._neq = parameters["Neq"][0]
        self._nphys = len(parameters["physics"]["physical_variable_names"])
        self._ndim = parameters["Ndim"][0]
        self._n_partitions = n_partitions
        self._raw_equilibriums = raw_equilibriums
        self._raw_solution_boundary_infos = raw_solution_boundary_infos
        self._mesh = mesh

    def _store_raw_partitions(self, raw_solutions, raw_solutions_skeleton, raw_gradients):
        self._raw_solutions = []
        self._raw_solutions_skeleton = []
        self._raw_gradients = []

        for raw_solution, raw_solution_skeleton, raw_gradient in zip(
            raw_solutions, raw_solutions_skeleton, raw_gradients
        ):
            self._raw_solutions.append(raw_solution.reshape(raw_solution.shape[0] // self.neq, self.neq))
            self._raw_solutions_skeleton.append(
                raw_solution_skeleton.reshape(raw_solution_skeleton.shape[0] // self.neq, self.neq)
            )
            raw_gradient = raw_gradient.reshape(raw_gradient.shape[0] // (self.neq * self.ndim), self.neq * self.ndim)
            self._raw_gradients.append(raw_gradient.reshape(raw_gradient.shape[0], self.neq, self.ndim))

    def _initial_setup(self):
        self._init_state_containers()
        self._init_flags()
        self._init_variable_indices()
        self._init_charge_scale()
        self._normalize_external_heating_inputs()

    def _init_state_containers(self):
        self._views = SolutionViews()
        self._summary = SolutionSummaryState()
        self._parameter_state = ParameterState()
        self._atomic_rates = AtomicRateState()
        self._interpolators_state = InterpolatorState()

    def _init_flags(self):
        self._combined_simple_solution = False
        self._full_phys_initialized = False
        self._simple_phys_initialized = False
        self._combined_to_full = False
        self._combined_boundary = False

    def _init_variable_indices(self):
        self._cons_idx = {}
        for i,label in enumerate(self.parameters['physics']['conservative_variable_names']):
            self._cons_idx[label] = i
        self._phys_idx = {}
        for i,label in enumerate(self.parameters['physics']['physical_variable_names']):
            self._phys_idx[label] = i

    def _init_charge_scale(self):
        if 'charge_scale' in self.parameters['adimensionalization'].keys():
            self._e = self.parameters['adimensionalization']['charge_scale']
        else:
            self._e = 1.60217662e-19
            self.parameters['adimensionalization']['charge_scale'] = self.e

    def _normalize_external_heating_inputs(self):
        energy_scale = (
            self.parameters['adimensionalization']['specific_energy_density_scale']
            / self.parameters['adimensionalization']['time_scale']
            * self.parameters['adimensionalization']['mass_scale']
        )
        for key in ('external_heating', 'external_heating_i', 'external_heating_e'):
            if key in self.parameters['physics']:
                self.parameters['physics'][key] = self.parameters['physics'][key] * energy_scale


    @property
    def parameters(self):
        """Dictionary with solution parameters"""
        return self._parameters

    @property
    def atomic_parameters(self):
        """Dictionary with atomic parameters"""
        return self._parameter_state.atomic
    @atomic_parameters.setter
    def atomic_parameters(self,value):
        self._parameter_state.atomic = value

    @property
    def dnn_parameters(self):
        """Dictionary with atomic parameters"""
        return self._parameter_state.neutral_diffusion
    @dnn_parameters.setter
    def dnn_parameters(self,value):
        self._parameter_state.neutral_diffusion = value
        self._parameter_state.neutral_diffusion['dnn_max_adim']=(self._parameter_state.neutral_diffusion['dnn_max']/
                                              self.parameters['adimensionalization']['length_scale']**2*
                                              self.parameters['adimensionalization']['time_scale'])
        if not value['const']:
            self._parameter_state.neutral_diffusion['dnn_min_adim']=(self._parameter_state.neutral_diffusion['dnn_min']/
                                              self.parameters['adimensionalization']['length_scale']**2*
                                              self.parameters['adimensionalization']['time_scale'])
    @property
    def dk_parameters(self):
        """Dictionary with atomic parameters"""
        return self._parameter_state.turbulence
    @dk_parameters.setter
    def dk_parameters(self,value):
        self._parameter_state.turbulence = value
        self._parameter_state.turbulence['dk_max_adim']=(self._parameter_state.turbulence['dk_max']/
                                              self.parameters['adimensionalization']['length_scale']**2*
                                              self.parameters['adimensionalization']['time_scale'])
        
        self._parameter_state.turbulence['dk_min_adim']=(self._parameter_state.turbulence['dk_min']/
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
    def views(self):
        """Public structured access to view-based solution state."""
        return self._views

    @property
    def summary(self):
        """Public structured access to summary and conservation state."""
        return self._summary

    @property
    def parameter_state(self):
        """Public structured access to setup parameter state."""
        return self._parameter_state

    @property
    def atomic_rates(self):
        """Public structured access to cached atomic rate coefficients."""
        return self._atomic_rates

    @property
    def interpolators(self):
        """Public structured access to cached interpolators."""
        return self._interpolators_state

    def recombine_full_solution(self):
        """ 
        Recombine raw solutions into one single mesh
        """
        assembly_ops.recombine_full_solution(self)

    def recombine_simple_full_solution(self):
        """
        Obtain the solution and gradient in a size the same as the vertices
        For the repeating vertices only one (we do not actually now which) value is saved
        This routine is useful for simple overview plots
        """
        assembly_ops.recombine_simple_full_solution(self)
    
    def recombine_boundary_solution(self):
        """
        extracts solutions and its gradients on the boundary
        """
        assembly_ops.recombine_boundary_solution(self)
        
    
    def calculate_in_gauss_points(self):
        assembly_ops.calculate_in_gauss_points(self)
    
    def calculate_in_boundary_gauss_points(self,boundaries):
        return assembly_ops.calculate_in_boundary_gauss_points(self, boundaries)

    def summary_along_the_wall(self):
        return boundary_ops.summary_along_the_wall(self)



        

        



    def plot_overview(self,n_levels=100):
        return plotting_ops.plot_overview(self, n_levels=n_levels)
        
    def plot_overview_difference(self,second_solution,n_levels=100):
        return plotting_ops.plot_overview_difference(self, second_solution, n_levels=n_levels)
    
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

        
        physical_ops.init_phys_variables(self, which=which)



        
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
        physical_ops.cons2phys(self, data)
                                                    
    def plot_overview_physical(self,n_levels=100, limits=None,ticks=None):
            return plotting_ops.plot_overview_physical(self, n_levels=n_levels, limits=limits, ticks=ticks)


    def plot_overview_physical_difference(self,second_solution,n_levels=100):
            return plotting_ops.plot_overview_physical_difference(self, second_solution, n_levels=n_levels)

    def plot_variables_overview(self,variable_list,labels,limits,n_levels,ticks,tick_lables,logs,title=None):
        return plotting_ops.plot_variables_overview(
            self, variable_list, labels, limits, n_levels, ticks, tick_lables, logs, title=title
        )


    def define_magnetic_axis(self):
        """
        defines magnetic axis as minimum of psi
        """
        equilibrium_ops.define_magnetic_axis(self)
    
    def define_minor_radii(self,which='simple'):
        """
        calculates minor radii either with given magnetic axis
        """
        equilibrium_ops.define_minor_radii(self, which=which)


    def define_qcyl(self,which='simple'):
        """
        calculates cylindrical safety factor radii either with given magnetic axis
        """
        equilibrium_ops.define_qcyl(self, which=which)




        

    def calculate_variables_along_line(self,r_line,z_line,variable_list):
        """
        calculates plasma parameters noted in variables list on a given line
        returns a dictionary with variables as keys and values along lines for them
        """
        return sampling_ops.calculate_variables_along_line(self, r_line, z_line, variable_list)



    def save_summary_line(self,save_folder,r_line,z_line,variable_list):


        return sampling_ops.save_summary_line(self, save_folder, r_line, z_line, variable_list)

    



    def calculate_ohmic_source(self, which='simple'):
        return plasma_source_ops.calculate_ohmic_source(self, which)
            
            


        
    def calculate_power_balance(self):
        return analysis_ops.calculate_power_balance(self)

    def calculate_volumetric_sources(self):
        return analysis_ops.calculate_volumetric_sources(self)
        
    def calculate_power_losses_to_wall(self):
        return analysis_ops.calculate_power_losses_to_wall(self)

    def calculate_boundary_summary(self):
        return boundary_ops.calculate_boundary_summary(self)
        
        




    def calculate_ionization_rate(self,which="simple"):
        return plasma_source_ops.calculate_ionization_rate(self, which)

    def calculate_recombination_rate(self,which="simple"):
        return plasma_source_ops.calculate_recombination_rate(self, which)

    def calculate_cx_rate(self,which="simple"):
        return plasma_source_ops.calculate_cx_rate(self, which)

    def calculate_dnn(self,which="simple"):
        return neutrals_ops.calculate_dnn(self, which)

    def calculate_dnn_with_nn_collision(self,which="simple"):
        return neutrals_ops.calculate_dnn_with_nn_collision(self, which)

    def calculate_dk(self,which="simple"):
        return turbulent_model_ops.calculate_dk(self, which)

    
    def calculate_mfp(self,which="simple"):
        return neutrals_ops.calculate_mfp(self, which)


    def calculate_ionization_source(self,which="simple"):
        return plasma_source_ops.calculate_ionization_source(self, which)
    def calculate_ion_gain_due_to_iz(self,which="simple"):
        return plasma_source_ops.calculate_ion_gain_due_to_iz(self, which)

    def calculate_ion_sink_due_to_rec(self,which="simple"):
        return plasma_source_ops.calculate_ion_sink_due_to_rec(self, which)
    def calculate_ion_sink_due_to_cx(self,which="simple"):
        return plasma_source_ops.calculate_ion_sink_due_to_cx(self, which)
            
        
    def calculate_electron_sink_due_to_iz(self,which="simple"):
        return plasma_source_ops.calculate_electron_sink_due_to_iz(self, which)


    def calculate_electron_sink_due_to_rec(self,which="simple"):
        return plasma_source_ops.calculate_electron_sink_due_to_rec(self, which)
                                                                
    def calculate_electron_gain_due_to_rec(self,which="simple"):
        return plasma_source_ops.calculate_electron_gain_due_to_rec(self, which)
    def calculate_electron_sink_due_to_cooling_factor(self,which="simple"):
        return plasma_source_ops.calculate_electron_sink_due_to_cooling_factor(self, which)

    def calculate_cooling_factor(self,which="simple"):
        return plasma_source_ops.calculate_cooling_factor(self, which)
    def calculate_cx_source(self,which="simple"):
        return plasma_source_ops.calculate_cx_source(self, which)
    def define_interpolators(self):
        """
        defines interpolators for full solutions and gradients based on shape functions
        """

        sampling_ops.define_interpolators(self)

    def n(self,r,z):
        return pointwise_fields_ops.n(self,r,z)

    def ti(self,r,z):
        return pointwise_fields_ops.ti(self,r,z)

    def te(self,r,z):
        return pointwise_fields_ops.te(self,r,z)
    
    def u(self,r,z):
        return pointwise_fields_ops.u(self,r,z)
    
    def cs(self,r,z):
        return pointwise_fields_ops.cs(self,r,z)
    
    def M(self,r,z):
        return pointwise_fields_ops.M(self,r,z)

    def nn(self,r,z):
        return pointwise_fields_ops.nn(self,r,z)

    def ionization_source_interp(self,r,z):
        return pointwise_fields_ops.ionization_source_interp(self,r,z)

    def iz_rate(self,r,z):
        return pointwise_fields_ops.iz_rate(self,r,z)

    def cx_rate(self,r,z):
        return pointwise_fields_ops.cx_rate(self,r,z)
    
    def dnn(self,r,z):
        return pointwise_fields_ops.dnn(self,r,z)
    def k(self,r,z):
        return pointwise_fields_ops.k(self,r,z)
    def dk(self,r,z):
        return pointwise_fields_ops.dk(self,r,z)
    
    def mfp_nn(self,r,z):
        return pointwise_fields_ops.mfp_nn(self,r,z)

    def p_dyn(self,r,z):
        return pointwise_fields_ops.p_dyn(self,r,z)
    def pi(self,r,z):
        return pointwise_fields_ops.pi(self,r,z)
    
    def grad_ti(self,r,z,coordinate):
        return pointwise_fields_ops.grad_ti(self,r,z,coordinate)

    def grad_pi(self,r,z,coordinate):
        return pointwise_fields_ops.grad_pi(self,r,z,coordinate)

    def grad_ti_par(self,r,z):
        return pointwise_fields_ops.grad_ti_par(self,r,z)

    def grad_te(self,r,z,coordinate):
        return pointwise_fields_ops.grad_te(self,r,z,coordinate)

    def grad_te_par(self,r,z):
        return pointwise_fields_ops.grad_te_par(self,r,z)

    def particle_flux_par(self,r,z):
        return pointwise_fields_ops.particle_flux_par(self,r,z)
    
    def ion_heat_flux_par_conv(self,r,z):
        return pointwise_fields_ops.ion_heat_flux_par_conv(self,r,z)


    def ion_heat_flux_par_cond(self,r,z):
        return pointwise_fields_ops.ion_heat_flux_par_cond(self,r,z)


    
    def ion_heat_flux_par(self,r,z):
        return pointwise_fields_ops.ion_heat_flux_par(self,r,z)

    def electron_heat_flux_par_conv(self,r,z):
        return pointwise_fields_ops.electron_heat_flux_par_conv(self,r,z)

    def electron_heat_flux_par_cond(self,r,z):
        return pointwise_fields_ops.electron_heat_flux_par_cond(self,r,z)


    def electron_heat_flux_par(self,r,z):
        return pointwise_fields_ops.electron_heat_flux_par(self,r,z)
    
    def psi(self,r,z):
        return pointwise_fields_ops.psi(self,r,z)

        
    
        
    def B(self,r,z,component):
        return pointwise_fields_ops.B(self,r,z,component)
    
    def grad_B(self,r,z,component,coordinate):
        return pointwise_fields_ops.grad_B(self,r,z,component,coordinate)

    def Q_e_loss_iz(self,r,z):
        return pointwise_fields_ops.Q_e_loss_iz(self,r,z)

    def Q_e_loss_rec(self,r,z):
        return pointwise_fields_ops.Q_e_loss_rec(self,r,z)

    def Q_e_gain_rec(self,r,z):
        return pointwise_fields_ops.Q_e_gain_rec(self,r,z)

    def Q_e_loss_tot(self,r,z):
        return pointwise_fields_ops.Q_e_loss_tot(self,r,z)

    def Q_i_gain_iz(self,r,z):
        return pointwise_fields_ops.Q_i_gain_iz(self,r,z)

    def Q_i_loss_rec(self,r,z):
        return pointwise_fields_ops.Q_i_loss_rec(self,r,z)

    def Q_i_loss_cx(self,r,z):
        return pointwise_fields_ops.Q_i_loss_cx(self,r,z)

    def Q_i_loss_tot(self,r,z):
        return pointwise_fields_ops.Q_i_loss_tot(self,r,z)


    def Q_loss_tot(self,r,z):
        return pointwise_fields_ops.Q_loss_tot(self,r,z)


attach_compat_properties(HDGsolution)
