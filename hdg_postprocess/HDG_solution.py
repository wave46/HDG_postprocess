import numpy as np
import matplotlib.pyplot as plt
import hdg_postprocess.solution_operations.pointwise_fields as pointwise_fields_impl
from hdg_postprocess.routines.atomic import *
from hdg_postprocess.routines.plasma import *
from hdg_postprocess.routines.neutrals import *
from raysect.core.math.function.float import Discrete2DMesh
from hdg_postprocess.routines.interpolators import SoledgeHDG2DInterpolator
from hdg_postprocess.view_containers import (
    AtomicRateState,
    InterpolatorState,
    ParameterState,
    SolutionSummaryState,
    SolutionViews,
)
from hdg_postprocess.solution_operations import (
    calculate_boundary_summary as calculate_boundary_summary_impl,
    calculate_dnn as calculate_dnn_impl,
    calculate_dnn_with_nn_collision as calculate_dnn_with_nn_collision_impl,
    calculate_dk as calculate_dk_impl,
    calculate_cooling_factor as calculate_cooling_factor_impl,
    calculate_cx_rate as calculate_cx_rate_impl,
    calculate_cx_source as calculate_cx_source_impl,
    calculate_electron_gain_due_to_rec as calculate_electron_gain_due_to_rec_impl,
    calculate_electron_sink_due_to_cooling_factor as calculate_electron_sink_due_to_cooling_factor_impl,
    calculate_electron_sink_due_to_iz as calculate_electron_sink_due_to_iz_impl,
    calculate_electron_sink_due_to_rec as calculate_electron_sink_due_to_rec_impl,
    calculate_in_boundary_gauss_points as calculate_in_boundary_gauss_points_impl,
    calculate_in_gauss_points as calculate_in_gauss_points_impl,
    calculate_ion_gain_due_to_iz as calculate_ion_gain_due_to_iz_impl,
    calculate_ion_sink_due_to_cx as calculate_ion_sink_due_to_cx_impl,
    calculate_ion_sink_due_to_rec as calculate_ion_sink_due_to_rec_impl,
    calculate_ionization_rate as calculate_ionization_rate_impl,
    calculate_ionization_source as calculate_ionization_source_impl,
    calculate_mfp as calculate_mfp_impl,
    calculate_ohmic_source as calculate_ohmic_source_impl,
    plot_overview as plot_overview_impl,
    plot_overview_difference as plot_overview_difference_impl,
    plot_overview_physical as plot_overview_physical_impl,
    plot_overview_physical_difference as plot_overview_physical_difference_impl,
    plot_variables_overview as plot_variables_overview_impl,
    calculate_power_balance as calculate_power_balance_impl,
    calculate_power_losses_to_wall as calculate_power_losses_to_wall_impl,
    calculate_variables_along_line as calculate_variables_along_line_impl,
    calculate_volumetric_sources as calculate_volumetric_sources_impl,
    cons2phys as cons2phys_impl,
    define_interpolators as define_interpolators_impl,
    define_magnetic_axis as define_magnetic_axis_impl,
    define_minor_radii as define_minor_radii_impl,
    define_qcyl as define_qcyl_impl,
    init_phys_variables as init_phys_variables_impl,
    calculate_recombination_rate as calculate_recombination_rate_impl,
    recombine_boundary_solution as recombine_boundary_solution_impl,
    recombine_full_solution as recombine_full_solution_impl,
    recombine_simple_full_solution as recombine_simple_full_solution_impl,
    save_summary_line as save_summary_line_impl,
    summary_along_the_wall as summary_along_the_wall_impl,
)
import os


def _resolve_nested_attr(obj, path):
    value = obj
    for part in path:
        value = getattr(value, part)
    return value


def _make_generated_property(root_attr, path, doc, setter_attr=None):
    def getter(self):
        base = self if not root_attr else getattr(self, root_attr)
        return _resolve_nested_attr(base, path)

    getter_parts = tuple(part for part in (root_attr.strip("_"),) + tuple(path) if part)
    getter.__name__ = f"get_{'_'.join(getter_parts)}"

    if setter_attr is None:
        return property(getter, doc=doc)

    def setter(self, value):
        setattr(self, setter_attr, value)

    setter.__name__ = f"set_{'_'.join(getter_parts)}"
    return property(getter, setter, doc=doc)


class HDGsolution:
    ""
    _AUX_CONTAINER_PATHS = {
        "_atomic_parameters": ("_parameter_state", "atomic"),
        "_dnn_parameters": ("_parameter_state", "neutral_diffusion"),
        "_dk_parameters": ("_parameter_state", "turbulence"),
        "_ionization_rate_simple": ("_atomic_rates", "ionization_simple"),
        "_recombination_rate_simple": ("_atomic_rates", "recombination_simple"),
        "_cx_rate_simple": ("_atomic_rates", "cx_simple"),
        "_sample_interpolator": ("_interpolators_state", "sample"),
        "_solution_interpolators": ("_interpolators_state", "solution"),
        "_gradient_interpolators": ("_interpolators_state", "gradient"),
        "_field_interpolators": ("_interpolators_state", "field"),
        "_qcyl_interpolator": ("_interpolators_state", "qcyl"),
    }
    _VIEW_CONTAINER_PATHS = {
        "_solution_simple": ("simple", "solution", "conservative"),
        "_gradient_simple": ("simple", "gradient", "conservative"),
        "_solution_simple_phys": ("simple", "solution", "physical"),
        "_gradient_simple_phys": ("simple", "gradient", "physical"),
        "_solution_glob": ("glob", "solution", "conservative"),
        "_gradient_glob": ("glob", "gradient", "conservative"),
        "_solution_glob_phys": ("glob", "solution", "physical"),
        "_gradient_glob_phys": ("glob", "gradient", "physical"),
        "_solution_gauss": ("gauss", "solution", "conservative"),
        "_gradient_gauss": ("gauss", "gradient", "conservative"),
        "_solution_boundary": ("boundary", "solution", "conservative"),
        "_solution_skeleton_boundary": ("boundary", "solution_skeleton", "conservative"),
        "_gradient_boundary": ("boundary", "gradient", "conservative"),
        "_magnetic_field_simple": ("simple", "equilibrium", "magnetic_field"),
        "_jtor_simple": ("simple", "equilibrium", "jtor"),
        "_poloidal_flux_simple": ("simple", "equilibrium", "poloidal_flux"),
        "_a_simple": ("simple", "equilibrium", "a"),
        "_magnetic_field_glob": ("glob", "equilibrium", "magnetic_field"),
        "_magnetic_field_unit_glob": ("glob", "equilibrium", "magnetic_field_unit"),
        "_jtor_glob": ("glob", "equilibrium", "jtor"),
        "_poloidal_flux_glob": ("glob", "equilibrium", "poloidal_flux"),
        "_a_glob": ("glob", "equilibrium", "a"),
        "_qcyl_simple": ("simple", "equilibrium", "qcyl"),
        "_qcyl_glob": ("glob", "equilibrium", "qcyl"),
        "_magnetic_field_gauss": ("gauss", "equilibrium", "magnetic_field"),
        "_magnetic_field_unit_gauss": ("gauss", "equilibrium", "magnetic_field_unit"),
        "_jtor_gauss": ("gauss", "equilibrium", "jtor"),
        "_poloidal_flux_gauss": ("gauss", "equilibrium", "poloidal_flux"),
        "_magnetic_field_boundary": ("boundary", "equilibrium", "magnetic_field"),
        "_magnetic_field_unit_boundary": ("boundary", "equilibrium", "magnetic_field_unit"),
        "_poloidal_flux_boundary": ("boundary", "equilibrium", "poloidal_flux"),
        "_solution_boundary_gauss": ("boundary_gauss", "solution", "conservative"),
        "_solution_skeleton_boundary_gauss": ("boundary_gauss", "solution_skeleton", "conservative"),
        "_gradient_boundary_gauss": ("boundary_gauss", "gradient", "conservative"),
        "_magnetic_field_boundary_gauss": ("boundary_gauss", "equilibrium", "magnetic_field"),
        "_magnetic_field_unit_boundary_gauss": ("boundary_gauss", "equilibrium", "magnetic_field_unit"),
        "_poloidal_flux_boundary_gauss": ("boundary_gauss", "equilibrium", "poloidal_flux"),
        "_dnn_simple": ("simple", "derived", "dnn"),
        "_dnn_simple_with_nn_collision": ("glob", "derived", "dnn_with_nn_collision"),
        "_dnn_simple_with_nn_collision_simple": ("simple", "derived", "dnn_with_nn_collision"),
        "_dk_simple": ("simple", "derived", "dk"),
        "_dk_glob": ("glob", "derived", "dk"),
        "_mfp_simple": ("simple", "derived", "mfp"),
        "_ionization_source": ("glob", "sources", "ionization_source"),
        "_ionization_source_simple": ("simple", "sources", "ionization_source"),
        "_ionization_source_gauss": ("gauss", "sources", "ionization_source"),
        "_ion_gain_iz": ("glob", "sources", "ion_gain_iz"),
        "_ion_gain_iz_simple": ("simple", "sources", "ion_gain_iz"),
        "_ion_gain_iz_gauss": ("gauss", "sources", "ion_gain_iz"),
        "_ion_sink_rec": ("glob", "sources", "ion_sink_rec"),
        "_ion_sink_rec_simple": ("simple", "sources", "ion_sink_rec"),
        "_ion_sink_rec_gauss": ("gauss", "sources", "ion_sink_rec"),
        "_ion_sink_cx": ("glob", "sources", "ion_sink_cx"),
        "_ion_sink_cx_simple": ("simple", "sources", "ion_sink_cx"),
        "_ion_sink_cx_gauss": ("gauss", "sources", "ion_sink_cx"),
        "_electron_sink_iz": ("glob", "sources", "electron_sink_iz"),
        "_electron_sink_iz_simple": ("simple", "sources", "electron_sink_iz"),
        "_electron_sink_iz_gauss": ("gauss", "sources", "electron_sink_iz"),
        "_electron_sink_rec": ("glob", "sources", "electron_sink_rec"),
        "_electron_sink_rec_simple": ("simple", "sources", "electron_sink_rec"),
        "_electron_sink_rec_gauss": ("gauss", "sources", "electron_sink_rec"),
        "_electron_gain_rec": ("glob", "sources", "electron_gain_rec"),
        "_electron_gain_rec_simple": ("simple", "sources", "electron_gain_rec"),
        "_electron_gain_rec_gauss": ("gauss", "sources", "electron_gain_rec"),
        "_electron_sink_cooling_factor": ("glob", "sources", "electron_sink_cooling_factor"),
        "_electron_sink_cooling_factor_simple": ("simple", "sources", "electron_sink_cooling_factor"),
        "_electron_sink_cooling_factor_gauss": ("gauss", "sources", "electron_sink_cooling_factor"),
        "_cooling_factor": ("glob", "sources", "cooling_factor"),
        "_cooling_factor_simple": ("simple", "sources", "cooling_factor"),
        "_cooling_factor_gauss": ("gauss", "sources", "cooling_factor"),
        "_cx_source": ("glob", "sources", "cx_source"),
        "_cx_source_simple": ("simple", "sources", "cx_source"),
        "_cx_source_gauss": ("gauss", "sources", "cx_source"),
        "_external_heating": ("glob", "sources", "external_heating"),
        "_external_heating_simple": ("simple", "sources", "external_heating"),
        "_external_heating_gauss": ("gauss", "sources", "external_heating"),
        "_external_heating_e": ("glob", "sources", "external_heating_e"),
        "_external_heating_e_simple": ("simple", "sources", "external_heating_e"),
        "_external_heating_e_gauss": ("gauss", "sources", "external_heating_e"),
        "_external_heating_i": ("glob", "sources", "external_heating_i"),
        "_external_heating_i_simple": ("simple", "sources", "external_heating_i"),
        "_external_heating_i_gauss": ("gauss", "sources", "external_heating_i"),
        "_ohmic_source": ("glob", "sources", "ohmic_source"),
        "_ohmic_source_simple": ("simple", "sources", "ohmic_source"),
        "_ohmic_source_gauss": ("gauss", "sources", "ohmic_source"),
    }
    _SUMMARY_CONTAINER_PATHS = {
        "_r_axis": ("equilibrium", "axis", "r"),
        "_z_axis": ("equilibrium", "axis", "z"),
        "_ion_gain_iz_total": ("sources", "ion_gain_iz_total"),
        "_ion_sink_rec_total": ("sources", "ion_sink_rec_total"),
        "_ion_sink_cx_total": ("sources", "ion_sink_cx_total"),
        "_electron_sink_iz_total": ("sources", "electron_sink_iz_total"),
        "_electron_sink_rec_total": ("sources", "electron_sink_rec_total"),
        "_electron_gain_rec_total": ("sources", "electron_gain_rec_total"),
        "_electron_sink_cooling_factor_total": ("sources", "electron_sink_cooling_factor_total"),
        "_external_heating_total": ("sources", "external_heating_total"),
        "_external_heating_e_total": ("sources", "external_heating_e_total"),
        "_external_heating_i_total": ("sources", "external_heating_i_total"),
        "_ohmic_source_total": ("sources", "ohmic_source_total"),
        "_boundary_summary": ("boundary", "boundary_summary"),
        "_ion_energy_sheath_loss_total": ("boundary", "ion_energy_sheath_loss_total"),
        "_electron_energy_sheath_loss_total": ("boundary", "electron_energy_sheath_loss_total"),
    }
    
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

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        views = self.__dict__.get("_views")
        view_path = self._VIEW_CONTAINER_PATHS.get(name)
        if views is not None and view_path is not None:
            view_state = getattr(views, view_path[0])
            field_state = getattr(view_state, view_path[1])
            setattr(field_state, view_path[2], value)
        summary = self.__dict__.get("_summary")
        summary_path = self._SUMMARY_CONTAINER_PATHS.get(name)
        if summary is not None and summary_path is not None:
            summary_state = getattr(summary, summary_path[0])
            for part in summary_path[1:-1]:
                summary_state = getattr(summary_state, part)
            setattr(summary_state, summary_path[-1], value)
        aux_path = self._AUX_CONTAINER_PATHS.get(name)
        if aux_path is not None:
            aux_state = self.__dict__.get(aux_path[0])
            if aux_state is not None:
                setattr(aux_state, aux_path[1], value)
        return

    def _initial_setup(self):
        # simple representation of a solution
        self._combined_simple_solution = False
        self._views = SolutionViews()
        self._summary = SolutionSummaryState()
        self._parameter_state = ParameterState()
        self._atomic_rates = AtomicRateState()
        self._interpolators_state = InterpolatorState()
        #physical solution flags
        self._full_phys_initialized = False
        self._simple_phys_initialized = False
        mapped_state_names = (
            set(self._VIEW_CONTAINER_PATHS)
            | set(self._SUMMARY_CONTAINER_PATHS)
        )
        for name in sorted(mapped_state_names):
            setattr(self, name, None)
        for name in self._AUX_CONTAINER_PATHS:
            setattr(self, name, None)

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

        if 'external_heating' in self.parameters['physics'].keys():
            self.parameters['physics']['external_heating'] = (self.parameters['physics']['external_heating']*self.parameters['adimensionalization']['specific_energy_density_scale']/
                                        self.parameters['adimensionalization']['time_scale']*self.parameters['adimensionalization']['mass_scale'])
        if 'external_heating_i' in self.parameters['physics'].keys():
            self.parameters['physics']['external_heating_i'] = (self.parameters['physics']['external_heating_i']*self.parameters['adimensionalization']['specific_energy_density_scale']/
                                        self.parameters['adimensionalization']['time_scale']*self.parameters['adimensionalization']['mass_scale'])
        if 'external_heating_e' in self.parameters['physics'].keys():
            self.parameters['physics']['external_heating_e'] = (self.parameters['physics']['external_heating_e']*self.parameters['adimensionalization']['specific_energy_density_scale']/
                                        self.parameters['adimensionalization']['time_scale']*self.parameters['adimensionalization']['mass_scale'])

        self._combined_to_full = False
        self._combined_boundary = False


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
        self._atomic_parameters = value

    @property
    def dnn_parameters(self):
        """Dictionary with atomic parameters"""
        return self._parameter_state.neutral_diffusion
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
        return self._parameter_state.turbulence
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
        recombine_full_solution_impl(self)

    def recombine_simple_full_solution(self):
        """
        Obtain the solution and gradient in a size the same as the vertices
        For the repeating vertices only one (we do not actually now which) value is saved
        This routine is useful for simple overview plots
        """
        recombine_simple_full_solution_impl(self)
    
    def recombine_boundary_solution(self):
        """
        extracts solutions and its gradients on the boundary
        """
        recombine_boundary_solution_impl(self)
        
    
    def calculate_in_gauss_points(self):
        calculate_in_gauss_points_impl(self)
    
    def calculate_in_boundary_gauss_points(self,boundaries):
        return calculate_in_boundary_gauss_points_impl(self, boundaries)

    def summary_along_the_wall(self):
        return summary_along_the_wall_impl(self)



        

        



    def plot_overview(self,n_levels=100):
        return plot_overview_impl(self, n_levels=n_levels)
        
    def plot_overview_difference(self,second_solution,n_levels=100):
        return plot_overview_difference_impl(self, second_solution, n_levels=n_levels)
    
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

        
        init_phys_variables_impl(self, which=which)



        
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
        cons2phys_impl(self, data)
                                                    
    def plot_overview_physical(self,n_levels=100, limits=None,ticks=None):
            return plot_overview_physical_impl(self, n_levels=n_levels, limits=limits, ticks=ticks)


    def plot_overview_physical_difference(self,second_solution,n_levels=100):
            return plot_overview_physical_difference_impl(self, second_solution, n_levels=n_levels)

    def plot_variables_overview(self,variable_list,labels,limits,n_levels,ticks,tick_lables,logs,title=None):
        return plot_variables_overview_impl(
            self, variable_list, labels, limits, n_levels, ticks, tick_lables, logs, title=title
        )


    def define_magnetic_axis(self):
        """
        defines magnetic axis as minimum of psi
        """
        define_magnetic_axis_impl(self)
    
    def define_minor_radii(self,which='simple'):
        """
        calculates minor radii either with given magnetic axis
        """
        define_minor_radii_impl(self, which=which)


    def define_qcyl(self,which='simple'):
        """
        calculates cylindrical safety factor radii either with given magnetic axis
        """
        define_qcyl_impl(self, which=which)




        

    def calculate_variables_along_line(self,r_line,z_line,variable_list):
        """
        calculates plasma parameters noted in variables list on a given line
        returns a dictionary with variables as keys and values along lines for them
        """
        return calculate_variables_along_line_impl(self, r_line, z_line, variable_list)



    def save_summary_line(self,save_folder,r_line,z_line,variable_list):


        return save_summary_line_impl(self, save_folder, r_line, z_line, variable_list)

    



    def calculate_ohmic_source(self, which='simple'):
        return calculate_ohmic_source_impl(self, which)
            
            


        
    def calculate_power_balance(self):
        return calculate_power_balance_impl(self)

    def calculate_volumetric_sources(self):
        return calculate_volumetric_sources_impl(self)
        
    def calculate_power_losses_to_wall(self):
        return calculate_power_losses_to_wall_impl(self)

    def calculate_boundary_summary(self):
        return calculate_boundary_summary_impl(self)
        
        




    def calculate_ionization_rate(self,which="simple"):
        return calculate_ionization_rate_impl(self, which)

    def calculate_recombination_rate(self,which="simple"):
        return calculate_recombination_rate_impl(self, which)

    def calculate_cx_rate(self,which="simple"):
        return calculate_cx_rate_impl(self, which)

    def calculate_dnn(self,which="simple"):
        return calculate_dnn_impl(self, which)

    def calculate_dnn_with_nn_collision(self,which="simple"):
        return calculate_dnn_with_nn_collision_impl(self, which)

    def calculate_dk(self,which="simple"):
        return calculate_dk_impl(self, which)

    
    def calculate_mfp(self,which="simple"):
        return calculate_mfp_impl(self, which)


    def calculate_ionization_source(self,which="simple"):
        return calculate_ionization_source_impl(self, which)
    def calculate_ion_gain_due_to_iz(self,which="simple"):
        return calculate_ion_gain_due_to_iz_impl(self, which)

    def calculate_ion_sink_due_to_rec(self,which="simple"):
        return calculate_ion_sink_due_to_rec_impl(self, which)
    def calculate_ion_sink_due_to_cx(self,which="simple"):
        return calculate_ion_sink_due_to_cx_impl(self, which)
            
        
    def calculate_electron_sink_due_to_iz(self,which="simple"):
        return calculate_electron_sink_due_to_iz_impl(self, which)


    def calculate_electron_sink_due_to_rec(self,which="simple"):
        return calculate_electron_sink_due_to_rec_impl(self, which)
                                                                
    def calculate_electron_gain_due_to_rec(self,which="simple"):
        return calculate_electron_gain_due_to_rec_impl(self, which)
    def calculate_electron_sink_due_to_cooling_factor(self,which="simple"):
        return calculate_electron_sink_due_to_cooling_factor_impl(self, which)

    def calculate_cooling_factor(self,which="simple"):
        return calculate_cooling_factor_impl(self, which)
    def calculate_cx_source(self,which="simple"):
        return calculate_cx_source_impl(self, which)
    def define_interpolators(self):
        """
        defines interpolators for full solutions and gradients based on shape functions
        """

        define_interpolators_impl(self)

    def n(self,r,z):
        return pointwise_fields_impl.n(self,r,z)

    def ti(self,r,z):
        return pointwise_fields_impl.ti(self,r,z)

    def te(self,r,z):
        return pointwise_fields_impl.te(self,r,z)
    
    def u(self,r,z):
        return pointwise_fields_impl.u(self,r,z)
    
    def cs(self,r,z):
        return pointwise_fields_impl.cs(self,r,z)
    
    def M(self,r,z):
        return pointwise_fields_impl.M(self,r,z)

    def nn(self,r,z):
        return pointwise_fields_impl.nn(self,r,z)

    def ionization_source_interp(self,r,z):
        return pointwise_fields_impl.ionization_source_interp(self,r,z)

    def iz_rate(self,r,z):
        return pointwise_fields_impl.iz_rate(self,r,z)

    def cx_rate(self,r,z):
        return pointwise_fields_impl.cx_rate(self,r,z)
    
    def dnn(self,r,z):
        return pointwise_fields_impl.dnn(self,r,z)
    def k(self,r,z):
        return pointwise_fields_impl.k(self,r,z)
    def dk(self,r,z):
        return pointwise_fields_impl.dk(self,r,z)
    
    def mfp_nn(self,r,z):
        return pointwise_fields_impl.mfp_nn(self,r,z)

    def p_dyn(self,r,z):
        return pointwise_fields_impl.p_dyn(self,r,z)
    def pi(self,r,z):
        return pointwise_fields_impl.pi(self,r,z)
    
    def grad_ti(self,r,z,coordinate):
        return pointwise_fields_impl.grad_ti(self,r,z,coordinate)

    def grad_pi(self,r,z,coordinate):
        return pointwise_fields_impl.grad_pi(self,r,z,coordinate)

    def grad_ti_par(self,r,z):
        return pointwise_fields_impl.grad_ti_par(self,r,z)

    def grad_te(self,r,z,coordinate):
        return pointwise_fields_impl.grad_te(self,r,z,coordinate)

    def grad_te_par(self,r,z):
        return pointwise_fields_impl.grad_te_par(self,r,z)

    def particle_flux_par(self,r,z):
        return pointwise_fields_impl.particle_flux_par(self,r,z)
    
    def ion_heat_flux_par_conv(self,r,z):
        return pointwise_fields_impl.ion_heat_flux_par_conv(self,r,z)


    def ion_heat_flux_par_cond(self,r,z):
        return pointwise_fields_impl.ion_heat_flux_par_cond(self,r,z)


    
    def ion_heat_flux_par(self,r,z):
        return pointwise_fields_impl.ion_heat_flux_par(self,r,z)

    def electron_heat_flux_par_conv(self,r,z):
        return pointwise_fields_impl.electron_heat_flux_par_conv(self,r,z)

    def electron_heat_flux_par_cond(self,r,z):
        return pointwise_fields_impl.electron_heat_flux_par_cond(self,r,z)


    def electron_heat_flux_par(self,r,z):
        return pointwise_fields_impl.electron_heat_flux_par(self,r,z)
    
    def psi(self,r,z):
        return pointwise_fields_impl.psi(self,r,z)

        
    
        
    def B(self,r,z,component):
        return pointwise_fields_impl.B(self,r,z,component)
    
    def grad_B(self,r,z,component,coordinate):
        return pointwise_fields_impl.grad_B(self,r,z,component,coordinate)

    def Q_e_loss_iz(self,r,z):
        return pointwise_fields_impl.Q_e_loss_iz(self,r,z)

    def Q_e_loss_rec(self,r,z):
        return pointwise_fields_impl.Q_e_loss_rec(self,r,z)

    def Q_e_gain_rec(self,r,z):
        return pointwise_fields_impl.Q_e_gain_rec(self,r,z)

    def Q_e_loss_tot(self,r,z):
        return pointwise_fields_impl.Q_e_loss_tot(self,r,z)

    def Q_i_gain_iz(self,r,z):
        return pointwise_fields_impl.Q_i_gain_iz(self,r,z)

    def Q_i_loss_rec(self,r,z):
        return pointwise_fields_impl.Q_i_loss_rec(self,r,z)

    def Q_i_loss_cx(self,r,z):
        return pointwise_fields_impl.Q_i_loss_cx(self,r,z)

    def Q_i_loss_tot(self,r,z):
        return pointwise_fields_impl.Q_i_loss_tot(self,r,z)


    def Q_loss_tot(self,r,z):
        return pointwise_fields_impl.Q_loss_tot(self,r,z)


_SOURCE_VIEW_DOCS = {
    "ionization_source": "Ionization source",
    "ion_gain_iz": "Ion energy sink due to ionization",
    "ion_sink_rec": "Ion energy sink due to recombination",
    "ion_sink_cx": "Ion energy sink due to charge exchange",
    "electron_sink_iz": "Electron energy sink due to ionization",
    "electron_sink_rec": "Electron energy sink due to recombination",
    "electron_gain_rec": "Electron energy source due to recombination",
    "electron_sink_cooling_factor": "Sink due to cooling factor",
    "cooling_factor": "Cooling factor",
    "cx_source": "Charge-exchange source",
    "external_heating": "External heating source",
    "external_heating_e": "External heating source on electrons",
    "external_heating_i": "External heating source on ions",
    "ohmic_source": "Ohmic heating source",
}

_STATE_PROPERTY_DOCS = {
    "solution_simple": ("_views", ("simple", "solution", "conservative"), "solution simply united on a single mesh (means not taking into account repeating points) [Nvertices x neq]"),
    "gradient_simple": ("_views", ("simple", "gradient", "conservative"), "gradient simply united on a single mesh (means not taking into account repeating points) [Nvertices x neq x ndim]"),
    "solution_simple_phys": ("_views", ("simple", "solution", "physical"), "physical solution simply united on a single mesh (means not taking into account repeating points) [Nvertices x nphys]"),
    "gradient_simple_phys": ("_views", ("simple", "gradient", "physical"), "phisical gradient simply united on a single mesh (means not taking into account repeating points) [Nvertices x nphys x ndim]"),
    "magnetic_field_simple": ("_views", ("simple", "equilibrium", "magnetic_field"), "magnetic field recombined united on a single mesh (means not taking into account repeating points) [Nvertices x 3]"),
    "jtor_simple": ("_views", ("simple", "equilibrium", "jtor"), "plasma current recombined united on a single mesh (means not taking into account repeating points) [Nvertices]"),
    "poloidal_flux_simple": ("_views", ("simple", "equilibrium", "poloidal_flux"), "poloidal flux recombined united on a single mesh (means not taking into account repeating points) [Nvertices x 3]"),
    "solution_boundary": ("_views", ("boundary", "solution", "conservative"), "conservative solution on faces of the boundary [Nextfaces x n_nodes_per_face x neq]"),
    "solution_skeleton_boundary": ("_views", ("boundary", "solution_skeleton", "conservative"), "conservative skeleton solution on faces of the boundary [Nextfaces x n_nodes_per_face x neq]"),
    "gradient_boundary": ("_views", ("boundary", "gradient", "conservative"), "gradient on a boundary [Nextfaces x n_nodes_per_face x neq x ndim]"),
    "magnetic_field_boundary": ("_views", ("boundary", "equilibrium", "magnetic_field"), "magnetic field recombined on a boundary. This one has shape [Nextfaces x n_nodes_per_face x 3]"),
    "magnetic_field_unit_boundary": ("_views", ("boundary", "equilibrium", "magnetic_field_unit"), "magnetic field unit vector recombined on a boundary. This one has shape [Nextfaces x n_nodes_per_face x 3]"),
    "poloidal_flux_boundary": ("_views", ("boundary", "equilibrium", "poloidal_flux"), "poloidal flux recombined on a boundary. This one has shape [Nextfaces x n_nodes_per_face]"),
    "solution_boundary_gauss": ("_views", ("boundary_gauss", "solution", "conservative"), "conservative solution on gauss points of faces of the boundary [Nextfaces x n_gauss_points_per_face x neq]"),
    "solution_skeleton_boundary_gauss": ("_views", ("boundary_gauss", "solution_skeleton", "conservative"), "conservative skeleton solution on gauss points of faces of the boundary [Nextfaces x n_gauss_points_per_face x neq]"),
    "gradient_boundary_gauss": ("_views", ("boundary_gauss", "gradient", "conservative"), "gradient united on gauss points of faces of the boundary (means not taking into account repeating points) [Nextfaces x n_gauss_points_per_face x neq x ndim]"),
    "magnetic_field_boundary_gauss": ("_views", ("boundary_gauss", "equilibrium", "magnetic_field"), "magnetic field recombined on gauss points of faces of the boundary. This one has shape [Nextfaces x n_gauss_points_per_face x 3]"),
    "magnetic_field_unit_boundary_gauss": ("_views", ("boundary_gauss", "equilibrium", "magnetic_field_unit"), "magnetic field unit vector recombined on gauss points of faces of the boundary. This one has shape [Nextfaces x n_nodes_per_face x 3]"),
    "poloidal_flux_boundary_gauss": ("_views", ("boundary_gauss", "equilibrium", "poloidal_flux"), "poloidal flux recombined on gauss points of faces of the boundary. This one has shape [Nextfaces x n_gauss_points_per_face]"),
    "combined_to_full": ("", ("_combined_to_full",), "Flag which tells if the solution has been combined to full"),
    "combined_boundary": ("", ("_combined_boundary",), "Flag which tells if the solution has been combined on a boundary of mesh"),
    "combined_simple_solution": ("", ("_combined_simple_solution",), "Flag which tells if the solution has been combined to simple one on full mesh"),
    "full_phys_initialized": ("", ("_full_phys_initialized",), "Flag which tells if physical values has been initialized (full)"),
    "simple_phys_initialized": ("", ("_simple_phys_initialized",), "Flag which tells if physical values has been initialized (simple)"),
    "solution_glob": ("_views", ("glob", "solution", "conservative"), "Solution recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x neq]"),
    "gradient_glob": ("_views", ("glob", "gradient", "conservative"), "Gradients recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x neq x ndim]"),
    "magnetic_field_glob": ("_views", ("glob", "equilibrium", "magnetic_field"), "magnetic field recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x 3]"),
    "poloidal_flux_glob": ("_views", ("glob", "equilibrium", "poloidal_flux"), "poloidal flux recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x 3]"),
    "poloidal_flux_gauss": ("_views", ("gauss", "equilibrium", "poloidal_flux"), "poloidal flux recombined on a full mesh and calculated in gauss points. This one has shape [Nelems x gauss_points_per_elem]"),
    "magnetic_field_unit_glob": ("_views", ("glob", "equilibrium", "magnetic_field_unit"), "magnetic field unit vector recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x 3]"),
    "jtor_glob": ("_views", ("glob", "equilibrium", "jtor"), "plassma current recombined on a full mesh. This one has shape [Nelems x nodes_per_elem]"),
    "solution_gauss": ("_views", ("gauss", "solution", "conservative"), "Solution recombined on a full mesh and calculated in gauss points. This one has shape [Nelems x gauss_points_per_elem x neq]"),
    "gradient_gauss": ("_views", ("gauss", "gradient", "conservative"), "Gradients recombined on a full mesh and calculated in gauss points. This one has shape [Nelems x gauss_points_per_elem x neq x ndim]"),
    "magnetic_field_gauss": ("_views", ("gauss", "equilibrium", "magnetic_field"), "magnetic field recombined on a full mesh and calculated in gauss points. This one has shape [Nelems x gauss_points_per_elem x 3]"),
    "magnetic_field_unit_gauss": ("_views", ("gauss", "equilibrium", "magnetic_field_unit"), "magnetic field recombined on a full mesh and calculated in gauss points. This one has shape [Nelems x gauss_points_per_elem x 3]"),
    "jtor_gauss": ("_views", ("gauss", "equilibrium", "jtor"), "plassma current recombined on a full mesh and calculated in gauss points. This one has shape [Nelems x gauss_points_per_elem]"),
    "solution_glob_phys": ("_views", ("glob", "solution", "physical"), "Physical solution recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x nphys]"),
    "gradient_glob_phys": ("_views", ("glob", "gradient", "physical"), "Physical gradients recombined on a full mesh. This one has shape [Nelems x nodes_per_elem x nphys x ndim]"),
    "e": ("", ("_e",), "elemental_charge"),
    "cons_idx": ("", ("_cons_idx",), "dictionary with keys are the cons variables, values are the indexes o the corresponding equation"),
    "phys_idx": ("", ("_phys_idx",), "dictionary with keys are the phys variables, values are the indexes o the corresponding equation"),
    "r_axis": ("_summary", ("equilibrium", "axis", "r"), "R coordinate of magnetic axis"),
    "z_axis": ("_summary", ("equilibrium", "axis", "z"), "Z coordinate of magnetic axis"),
    "a_glob": ("_views", ("glob", "equilibrium", "a"), "minor radii on global mesh"),
    "a_simple": ("_views", ("simple", "equilibrium", "a"), "minor radii on simple mesh"),
    "qcyl_glob": ("_views", ("glob", "equilibrium", "qcyl"), "Cylindrical safety factor on global mesh"),
    "qcyl_simple": ("_views", ("simple", "equilibrium", "qcyl"), "Cylindrical safety factor on simple mesh"),
    "boundary_summary": ("_summary", ("boundary", "boundary_summary"), "A dictionary with boundary summary information"),
    "ion_energy_sheath_loss_total": ("_summary", ("boundary", "ion_energy_sheath_loss_total"), "Total ion energy loss in sheath on a full solution mesh using conservative values as inputs"),
    "electron_energy_sheath_loss_total": ("_summary", ("boundary", "electron_energy_sheath_loss_total"), "Total electron energy loss in sheath on a full solution mesh using conservative values as inputs"),
}

_SOURCE_VIEW_VARIANTS = {
    "": ("glob", "on a full solution mesh using conservative values as inputs"),
    "_simple": ("simple", "on a simple solution mesh"),
    "_gauss": ("gauss", "on gauss points"),
}

_SOURCE_TOTAL_DOCS = {
    "ion_gain_iz_total": "Total ion energy sink due to ionization on a full solution mesh using conservative values as inputs",
    "ion_sink_rec_total": "Total ion energy sink due to recombination on a full solution mesh using conservative values as inputs",
    "ion_sink_cx_total": "Total ion energy sink due to charge exchange on a full solution mesh using conservative values as inputs",
    "electron_sink_iz_total": "Total electron energy sink due to ionization on a full solution mesh using conservative values as inputs",
    "electron_sink_rec_total": "Total electron energy sink due to recombination on a full solution mesh using conservative values as inputs",
    "electron_gain_rec_total": "Total electron energy source due to recombination on a full solution mesh using conservative values as inputs",
    "external_heating_total": "Total external heating source on a full solution mesh using conservative values as inputs",
    "external_heating_e_total": "Total external heating source on electrons on a full solution mesh using conservative values as inputs",
    "external_heating_i_total": "Total external heating source on ions on a full solution mesh using conservative values as inputs",
    "ohmic_source_total": "Total ohmic heating source on a full solution mesh using conservative values as inputs",
}

_ATOMIC_RATE_DOCS = {
    "ionization_rate_simple": ("_atomic_rates", ("ionization_simple",), "Ionization rate coefficient on a simple solution mesh"),
    "recombination_rate_simple": ("_atomic_rates", ("recombination_simple",), "Recombination rate coefficient on a simple solution mesh"),
    "cx_rate_simple": ("_atomic_rates", ("cx_simple",), "Charge exchange rate coefficient on a simple solution mesh"),
}

_DERIVED_DOCS = {
    "dnn_simple": ("_views", ("simple", "derived", "dnn"), "Neutral diffusion on a simple solution mesh"),
    "dnn_simple_with_nn_collision": ("_views", ("glob", "derived", "dnn_with_nn_collision"), "Neutral diffusion with neutral-neutral diffusions on a global solution mesh"),
    "dnn_simple_with_nn_collision_simple": ("_views", ("simple", "derived", "dnn_with_nn_collision"), "Neutral diffusion with neutral-neutral diffusions on a simple solution mesh"),
    "dk_simple": ("_views", ("simple", "derived", "dk"), "Turbulent diffusion on a simple solution mesh"),
    "dk_glob": ("_views", ("glob", "derived", "dk"), "Turbulent diffusion on a full solution mesh"),
    "mfp_simple": ("_views", ("simple", "derived", "mfp"), "Neutral mean free path on a simple solution mesh"),
}

_INTERPOLATOR_DOCS = {
    "sample_interpolator": ("_interpolators_state", ("sample",), "Sample interpolator for acceleration", "_sample_interpolator"),
    "solution_interpolators": ("_interpolators_state", ("solution",), "A list of interpolators of solutions in conservative form", None),
    "gradient_interpolators": ("_interpolators_state", ("gradient",), "A list of interpolators of solutions in conservative form", None),
    "field_interpolators": ("_interpolators_state", ("field",), "A list of interpolators of magnetic field", None),
    "qcyl_interpolator": ("_interpolators_state", ("qcyl",), "A list of interpolators of magnetic field", None),
}


for _name, (_root_attr, _path, _doc) in _STATE_PROPERTY_DOCS.items():
    setattr(HDGsolution, _name, _make_generated_property(_root_attr, _path, _doc))

for _base_name, _doc_prefix in _SOURCE_VIEW_DOCS.items():
    for _suffix, (_view_name, _doc_suffix) in _SOURCE_VIEW_VARIANTS.items():
        setattr(
            HDGsolution,
            f"{_base_name}{_suffix}",
            _make_generated_property("_views", (_view_name, "sources", _base_name), f"{_doc_prefix} {_doc_suffix}"),
        )

for _name, _doc in _SOURCE_TOTAL_DOCS.items():
    setattr(HDGsolution, _name, _make_generated_property("_summary", ("sources", _name), _doc))

for _name, (_root_attr, _path, _doc) in _ATOMIC_RATE_DOCS.items():
    setattr(HDGsolution, _name, _make_generated_property(_root_attr, _path, _doc))

for _name, (_root_attr, _path, _doc) in _DERIVED_DOCS.items():
    setattr(HDGsolution, _name, _make_generated_property(_root_attr, _path, _doc))

for _name, (_root_attr, _path, _doc, _setter_attr) in _INTERPOLATOR_DOCS.items():
    setattr(HDGsolution, _name, _make_generated_property(_root_attr, _path, _doc, setter_attr=_setter_attr))
