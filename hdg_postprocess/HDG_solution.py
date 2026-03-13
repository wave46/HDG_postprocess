import numpy as np
import matplotlib.pyplot as plt
from hdg_postprocess.routines.atomic import *
from hdg_postprocess.routines.plasma import *
from hdg_postprocess.routines.neutrals import *
from raysect.core.math.function.float import Discrete2DMesh
from hdg_postprocess.routines.interpolators import SoledgeHDG2DInterpolator
from hdg_postprocess.solution_operations import (
    calculate_boundary_summary as calculate_boundary_summary_impl,
    calculate_dnn as calculate_dnn_impl,
    calculate_dnn_with_nn_collision as calculate_dnn_with_nn_collision_impl,
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
        self._jtor_simple = None
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
        self._ion_gain_iz_simple = None
        self._ion_sink_rec_simple = None
        self._ion_sink_cx_simple = None
        self._electron_sink_iz_simple = None
        self._electron_sink_rec_simple = None
        self._electron_gain_rec_simple = None
        self._electron_sink_cooling_factor_simple = None
        self._cooling_factor_simple = None
        self._cx_source_simple = None
        self._ionization_rate_simple = None   
        self._recombination_rate_simple = None        
        self._cx_rate_simple = None
        self._dnn_simple = None
        self._dnn_with_nn_collision_simple = None
        self._mfp_simple = None

        self._external_heating_simple = None
        self._external_heating_e_simple = None
        self._external_heating_i_simple = None
        

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
        self._poloidal_flux_gauss = None
        self._ohmic_source_gauss = None
        self._ionization_source_gauss = None
        self._ion_gain_iz_gauss = None
        self._ion_sink_rec_gauss = None
        self._ion_sink_cx_gauss = None
        self._electron_sink_iz_gauss = None
        self._electron_sink_rec_gauss = None
        self._electron_gain_rec_gauss = None
        self._electron_sink_cooling_factor_gauss = None
        self._cooling_factor_gauss = None
        self._cx_source_gauss = None
        self._external_heating_gauss = None
        self._external_heating_e_gauss = None
        self._external_heating_i_gauss = None

        # total volumetric values
        self._ohmic_source_total = None
        self._electron_sink_iz_total = None
        self._ion_gain_iz_total = None
        self._electron_sink_rec_total = None
        self._electron_gain_rec_total = None
        self._ion_sink_rec_total = None
        self._ion_sink_cx_total = None
        self._external_heating_total = None
        self._external_heating_e_total = None
        self._external_heating_i_total = None

        #boundary 
        self._boundary_summary = None
        self._ion_energy_sheath_loss_total = None
        self._electron_energy_sheath_loss_total = None

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


        if self._n_partitions == 1:
            #no need to recombine meshes
            self._combined_to_full = False
            self._combined_boundary = False
            self._solution_glob = None
            self._gradient_glob = None
            self._magnetic_field_glob = None
            self._magnetic_field_unit_glob = None
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
    def jtor_simple(self):
        """plasma current recombined united on a single mesh (means not taking into account repeating points) [Nvertices]"""
        return self._jtor_simple

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
    def poloidal_flux_boundary(self):
        """poloidal flux recombined on a boundary. This one has shape [Nextfaces x n_nodes_per_face]"""
        return self._poloidal_flux_boundary

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
    def poloidal_flux_boundary_gauss(self):
        """poloidal flux recombined on gauss points of faces of the boundary. This one has shape [Nextfaces x n_gauss_points_per_face]"""
        return self._poloidal_flux_boundary_gauss

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
    def poloidal_flux_gauss(self):
        """poloidal flux recombined on a full mesh and calculated in gauss points. This one has shape [Nelems x gauss_points_per_elem]"""
        return self._poloidal_flux_gauss

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
    def ion_gain_iz(self):
        """Ion energy sink due to ionization"""
        return self._ion_gain_iz

    @property
    def ion_gain_iz_simple(self):
        """Ion energy sink due to ionization on a simple solution mesh"""
        return self._ion_gain_iz_simple

    @property
    def ion_gain_iz_gauss(self):
        """Ion energy sink due to ionization on gauss points"""
        return self._ion_gain_iz_gauss
    
    @property
    def ion_gain_iz_total(self):
        """Total ion energy sink due to ionization on a full solution mesh using conservative values as inputs"""
        return self._ion_gain_iz_total
    
    @property
    def ion_sink_rec(self):
        """Ion energy sink due to recombination"""
        return self._ion_sink_rec
    
    @property
    def ion_sink_rec_simple(self):
        """Ion energy sink due to recombination on a simple solution mesh"""
        return self._ion_sink_rec_simple
    
    @property
    def ion_sink_rec_gauss(self):
        """Ion energy sink due to recombination on gauss points"""
        return self._ion_sink_rec_gauss
    
    @property
    def ion_sink_rec_total(self):
        """Total ion energy sink due to recombination on a full solution mesh using conservative values as inputs"""
        return self._ion_sink_rec_total
    
    @property
    def ion_sink_cx(self):
        """Ion energy sink due to charge exchange"""
        return self._ion_sink_cx
    
    @property
    def ion_sink_cx_simple(self):
        """Ion energy sink due to charge exchange on a simple solution mesh"""
        return self._ion_sink_cx_simple
    
    @property
    def ion_sink_cx_gauss(self):
        """Ion energy sink due to charge exchange on gauss points"""
        return self._ion_sink_cx_gauss
    
    @property
    def ion_sink_cx_total(self):
        """Total ion energy sink due to charge exchange on a full solution mesh using conservative values as inputs"""
        return self._ion_sink_cx_total

    @property
    def electron_sink_iz(self):
        """Electron energy sink due to ionization"""
        return self._electron_sink_iz
        
    @property
    def electron_sink_iz_simple(self):
        """Electron energy sink due to ionization on a simple solution mesh"""
        return self._electron_sink_iz_simple

    @property
    def electron_sink_iz_gauss(self):
        """Electron energy sink due to ionization on gauss points"""
        return self._electron_sink_iz_gauss
    
    @property
    def electron_sink_iz_total(self):
        """Total electron energy sink due to ionization on a full solution mesh using conservative values as inputs"""
        return self._electron_sink_iz_total

    @property
    def electron_sink_rec(self):
        """Electron energy sink due to recombination"""
        return self._electron_sink_rec

    @property
    def electron_sink_rec_simple(self):
        """Electron energy sink due to recombination on a simple solution mesh"""
        return self._electron_sink_rec_simple

    @property
    def electron_sink_rec_gauss(self):
        """Electron energy sink due to recombination on gauss points"""
        return self._electron_sink_rec_gauss
    
    @property
    def electron_sink_rec_total(self):
        """Total electron energy sink due to recombination on a full solution mesh using conservative values as inputs"""
        return self._electron_sink_rec_total

    @property
    def electron_gain_rec(self):
        """Ionization source on a simple solution mesh"""
        return self._electron_gain_rec

    @property
    def electron_gain_rec_simple(self):
        """Ionization source on a simple solution mesh"""
        return self._electron_gain_rec_simple

    @property
    def electron_gain_rec_gauss(self):
        """Ionization source on gauss points"""
        return self._electron_gain_rec_gauss

    @property
    def electron_gain_rec_total(self):
        """Total ionization source on a full solution mesh using conservative values as inputs"""
        return self._electron_gain_rec_total
    
    @property
    def electron_sink_cooling_factor(self):
        """Sink due to cooling factor on a solution mesh"""
        return self._electron_sink_cooling_factor

    @property
    def electron_sink_cooling_factor_simple(self):
        """Sink due to cooling factor on a simple solution mesh"""
        return self._electron_sink_cooling_factor_simple

    @property
    def electron_sink_cooling_factor_gauss(self):
        """Sink due to cooling factor on gauss points"""
        return self._electron_sink_cooling_factor_gauss
    
    @property
    def cooling_factor(self):
        """Cooling factor on a solution mesh"""
        return self._cooling_factor
    
    @property
    def cooling_factor_simple(self):
        """Cooling factor on a simple solution mesh"""
        return self._cooling_factor_simple
    
    @property
    def cooling_factor_gauss(self):
        """Cooling factor on gauss points"""
        return self._cooling_factor_gauss
    


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
    def external_heating(self):
        """External heating source on a full solution mesh using conservative values as inputs"""
        return self._external_heating
    @property
    def external_heating_simple(self):
        """External heating source on a simple solution mesh"""
        return self._external_heating_simple
    @property
    def external_heating_gauss(self):
        """External heating source on gauss points"""
        return self._external_heating_gauss
    @property
    def external_heating_total(self):
        """Total external heating source on a full solution mesh using conservative values as inputs"""
        return self._external_heating_total
    

    @property
    def external_heating_e(self):
        """External heating source on electrons on a full solution mesh using conservative values as inputs"""
        return self._external_heating_e
    @property
    def external_heating_e_simple(self):
        """External heating source on electrons on a simple solution mesh"""
        return self._external_heating_e_simple
    @property
    def external_heating_e_gauss(self): 
        """External heating source on electrons on gauss points"""
        return self._external_heating_e_gauss
    @property
    def external_heating_e_total(self):
        """Total external heating source on electrons on a full solution mesh using conservative values as inputs"""
        return self._external_heating_e_total

    @property
    def external_heating_i(self):
        """External heating source on ions on a full solution mesh using conservative values as inputs"""
        return self._external_heating_i
    @property
    def external_heating_i_simple(self):
        """External heating source on ions on a simple solution mesh"""
        return self._external_heating_i_simple
    @property
    def external_heating_i_gauss(self):
        """External heating source on ions on gauss points"""
        return self._external_heating_i_gauss
    @property
    def external_heating_i_total(self):
        """Total external heating source on ions on a full solution mesh using conservative values as inputs"""
        return self._external_heating_i_total

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
    def ohmic_source_total(self):
        """Total ohmic heating source on a full solution mesh using conservative values as inputs"""
        return self._ohmic_source_total
    
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

    @property
    def boundary_summary(self):
        """A dictionary with boundary summary information"""
        return self._boundary_summary

    @property
    def ion_energy_sheath_loss_total(self):
        """Total ion energy loss in sheath on a full solution mesh using conservative values as inputs"""
        return self._ion_energy_sheath_loss_total
    @property
    def electron_energy_sheath_loss_total(self):
        """Total electron energy loss in sheath on a full solution mesh using conservative values as inputs"""
        return self._electron_energy_sheath_loss_total
    



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
                                                          log=True,label=colorbar_labels[i],connectivity=self.mesh.connectivity_big,n_levels=n_levels,cmap='bwr')
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
            if (cons_variable != b'Gamma'):
                axes[i//2,i%2] = self.mesh.plot_full_mesh(difference_dimensional[:,i],ax=axes[i//2,i%2],
                                                          log=False,label=colorbar_labels[i],connectivity=self.mesh.connectivity_big,n_levels=n_levels)
            else:
                axes[i//2,i%2] = self.mesh.plot_full_mesh(difference_dimensional[:,i],ax=axes[i//2,i%2],
                                                          log=False,label=colorbar_labels[i],connectivity=self.mesh.connectivity_big,n_levels=n_levels,cmap='bwr')
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
            """
            Plot n, n_n, Ti, Te, M,k,....
            As a physical overview legacy
            """

            if not self._simple_phys_initialized:
                print('Initializing physical solution first')
                self.init_phys_variables('simple')



            colorbar_labels = [r'n [m$^{-3}$]',r'$n_n$ [m$^{-3}$]',r'$T_i [eV]$',r'$T_e [eV] $',r'M', r'$k$ [m$^2$/s$^2$]']
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
                if ticks == None:
                    tick = None
                else:
                    tick = ticks[i]
                if ((i!=4)and(i!=5)) :
                    data = solutions_plot[:,i].copy()
                    if (i == 0) or (i == 1):
                        data[data<0] = 1e8
                    else:
                        data[data<0] = 1e-3
                    axes[i//2,i%2] = self.mesh.plot_full_mesh(data,ax=axes[i//2,i%2],
                                                              log=True,label=colorbar_labels[i],connectivity=self.mesh.connectivity_big,n_levels=n_levels,limits=limit,ticks=tick)
                else:
                    data = solutions_plot[:,i].copy()
                    data[np.where(np.isnan(data))] = 0
                    if (i == 4):
                        axes[i//2,i%2] = self.mesh.plot_full_mesh(data,ax=axes[i//2,i%2],
                                                              log=False,label=colorbar_labels[i],connectivity=self.mesh.connectivity_big,n_levels=n_levels,limits=limit,cmap='bwr')
                    else:
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
                if (i==4):
                    axes[i//2,i%2] = self.mesh.plot_full_mesh(solutions_plot[:,i],ax=axes[i//2,i%2],
                                                              log=False,label=colorbar_labels[i],connectivity=self.mesh.connectivity_big,n_levels=n_levels,cmap='bwr')
                else:
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
            if variable == 'M':
                cmap = 'bwr'
            else:
                cmap = 'jet'
            if var_to_plot>2:
                axes[i//2,i%2] = self.mesh.plot_full_mesh(data,ax=axes[i//2,i%2],
                             log=log,label=label,connectivity=self.mesh.connectivity_big,n_levels=n_levels,
                             ticks=tick,tick_labels=tick_label,limits=limit,cmap=cmap)
            elif var_to_plot==2:
                axes[i%2] = self.mesh.plot_full_mesh(data,ax=axes[i%2],
                             log=log,label=label,connectivity=self.mesh.connectivity_big,n_levels=n_levels,
                             ticks=tick,tick_labels=tick_label,limits=limit,cmap=cmap)
            else:
                axes = self.mesh.plot_full_mesh(data,ax=axes,
                             log=log,label=label,connectivity=self.mesh.connectivity_big,n_levels=n_levels,
                             ticks=tick,tick_labels=tick_label,limits=limit,cmap=cmap)
        plt.tight_layout()
        return fig,axes,res


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
                                                 self.parameters['physics']['Mref'],
                                                 self._cons_idx)

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
    def k(self,r,z):
        """
        returns value of turbulent energy in given point (r,z)
        """
        if b'rho' not in self.parameters['physics']['physical_variable_names']:
            raise KeyError('density is not in the models')
        
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        #only fill needed field
        solution = np.zeros([1,self.neq])
        solution[:,self._cons_idx[b'k']] = self._solution_interpolators[self._cons_idx[b'k']](r,z)

        return calculate_k_cons(solution,self.parameters['adimensionalization']['k_scale'],self._cons_idx)
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

        return calculate_dk_cons(solution,self.dk_parameters, q_cyl,r/self.parameters['adimensionalization']['length_scale'],
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
    def pi(self,r,z):
        """
        returns value of ion pressure kbTi in given point (r,z)
        """

        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        solution = np.zeros([1,self.neq])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] ==0:
            return 0
        p0 =  (2/3/self.parameters['physics']['Mref'])*self.parameters['adimensionalization']['density_scale']* \
                           self.parameters['adimensionalization']['temperature_scale']*self.parameters['adimensionalization']['charge_scale']
        return calculate_pi_cons(solution, p0,self.cons_idx)
    
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

    def grad_pi(self,r,z,coordinate):
        """
        returns value of derivative of ion pressure over chosen direction in given point (r,z)
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
        p0 =  (2/3/self.parameters['physics']['Mref'])*self.parameters['adimensionalization']['density_scale']* \
                           self.parameters['adimensionalization']['temperature_scale']*self.parameters['adimensionalization']['charge_scale']
        return calculate_grad_pi_cons(solution,gradient,p0,self.parameters['adimensionalization']['length_scale'],self._cons_idx)[0][idx]

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
    
    def psi(self,r,z):
        """
        returns value of poloidal flux in given point (r,z)
        """
        
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        #only fill needed field
        psi = self._psi_interpolator(r,z)
        
        return psi

        
    
        
    def B(self,r,z,component):
        """
        returns value of one of the components of the magnetic field vector in given point (r,z)
        """
        if component == 'R':
            idx = 0
        elif component == 'Z':
            idx = 1
        elif component == 'theta':
            idx = 2
        else:
            raise ValueError(f'{component} is not a component of the problem')
        
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()

        return self._field_interpolators[idx](r,z)
    
    def grad_B(self,r,z,component,coordinate):
        """
        returns value of gradient in along given coordinate of
        one of the components of the magnetic field vector in given point (r,z)
        """

        if component == 'R':
            idx = 0
        elif component == 'Z':
            idx = 1
        elif component == 'theta':
            idx = 2
        else:
            raise ValueError(f'{component} is not a component of the problem')

        if coordinate == 'x':
            idx_grad = 0
        elif coordinate == 'y':
            idx_grad = 1
        else:
            raise ValueError(f'{coordinate} is not a coordinate of the problem')

        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()

        return self._field_interpolators[idx].gradient(r,z)[idx_grad]

    def Q_e_loss_iz(self,r,z):
        """
        returns value of electron energy loss due to ionization in given point (r,z)
        """
        if self.atomic_parameters is None:
            raise ValueError("Please, provide atomic settings for the simulation")
        if "Eiz" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide Eiz atomic settings for the simulation")
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        
        solution = np.zeros([1,self.neq])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] == 0:
            return 0
        return calculate_electron_sink_due_to_iz_cons(solution,self.atomic_parameters['Eiz'],
                                                 self.parameters['adimensionalization']['temperature_scale'],
                                                 self.parameters['adimensionalization']['density_scale'],
                                                 self.parameters['physics']['Mref'],
                                                 self.parameters['adimensionalization']['charge_scale'],
                                                 self._cons_idx)

    def Q_e_loss_rec(self,r,z):
        """
        returns value of electron energy loss due to recombination in given point (r,z)
        """
        if self.atomic_parameters is None:
            raise ValueError("Please, provide atomic settings for the simulation")
        if "Erec" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide Erec atomic settings for the simulation")
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        
        solution = np.zeros([1,self.neq])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] == 0:
            return 0
        return calculate_electron_sink_due_to_rec_cons(solution,self.atomic_parameters['Erec'],
                                                 self.parameters['adimensionalization']['temperature_scale'],
                                                 self.parameters['adimensionalization']['density_scale'],
                                                 self.parameters['physics']['Mref'],
                                                 self.parameters['adimensionalization']['charge_scale'])

    def Q_e_gain_rec(self,r,z):
        """
        returns value of electron energy gain due to recombination in given point (r,z)
        """
        if self.atomic_parameters is None:
            raise ValueError("Please, provide atomic settings for the simulation")
        if "rec" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide recombination atomic settings for the simulation")
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()

        solution = np.zeros([1,self.neq])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] == 0:
            return 0
        
        return calculate_electron_gain_due_to_rec_cons(solution,self.atomic_parameters['rec'],
                                                 self.parameters['adimensionalization']['temperature_scale'],
                                                 self.parameters['adimensionalization']['density_scale'],
                                                 self.parameters['physics']['Mref'],
                                                 self.parameters['adimensionalization']['charge_scale'])

    def Q_e_loss_tot(self,r,z):
        """
        returns value of total electron energy loss in given point (r,z)
        """
        if self.atomic_parameters is None:
            raise ValueError("Please, provide atomic settings for the simulation")
        if "Eiz" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide Eiz atomic settings for the simulation")
        if "Erec" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide Erec atomic settings for the simulation")
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()
        
        solution = np.zeros([1,self.neq])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] == 0:
            return 0

        return calculate_electron_total_loss_cons(solution,self.atomic_parameters['Eiz'],self.atomic_parameters['Erec'],
                                                    self.atomic_parameters['rec'],
                                                    self.parameters['adimensionalization']['temperature_scale'],
                                                    self.parameters['adimensionalization']['density_scale'],
                                                    self.parameters['physics']['Mref'],
                                                    self.parameters['adimensionalization']['charge_scale'],
                                                    self._cons_idx)

    def Q_i_gain_iz(self,r,z):
        """
        returns value of ion energy gain due to ionization in given point (r,z)
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
        if solution[0,0] == 0:
            return 0
        return calculate_ion_gain_due_to_iz_cons(solution,self.atomic_parameters['iz'],
                                                 self.parameters['adimensionalization']['temperature_scale'],
                                                 self.parameters['adimensionalization']['density_scale'],
                                                 self.parameters['physics']['Mref'],
                                                 self.parameters['physics']['R_E'],
                                                 self.parameters['adimensionalization']['charge_scale'],
                                                 self._cons_idx)

    def Q_i_loss_rec(self,r,z):
        """
        returns value of ion energy loss due to recombination in given point (r,z)
        """
        if self.atomic_parameters is None:
            raise ValueError("Please, provide atomic settings for the simulation")
        if "rec" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide recombination atomic settings for the simulation")
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()

        solution = np.zeros([1,self.neq])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] == 0:
            return 0
        return calculate_ion_sink_due_to_rec_cons(solution,self.atomic_parameters['rec'],
                                                 self.parameters['adimensionalization']['temperature_scale'],
                                                 self.parameters['adimensionalization']['density_scale'],
                                                 self.parameters['physics']['Mref'],
                                                 self.parameters['adimensionalization']['speed_scale']**2*self.parameters['adimensionalization']['mass_scale'])

    def Q_i_loss_cx(self,r,z):
        """
        returns value of ion energy loss due to charge exchange in given point (r,z)
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
        if solution[0,0] == 0:
            return 0
        return calculate_ion_sink_due_to_cx_cons(solution,self.atomic_parameters['cx'],
                                                 self.parameters['adimensionalization']['temperature_scale'],
                                                 self.parameters['adimensionalization']['density_scale'],
                                                 self.parameters['physics']['Mref'],
                                                 self.parameters['adimensionalization']['speed_scale'],
                                                 self.parameters['adimensionalization']['mass_scale'],
                                                 self._cons_idx)

    def Q_i_loss_tot(self,r,z):
        """
        returns value of total ion energy loss in given point (r,z)
        """
        if self.atomic_parameters is None:
            raise ValueError("Please, provide atomic settings for the simulation")
        if "iz" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide ionization atomic settings for the simulation")
        if "rec" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide recombination atomic settings for the simulation")
        if "cx" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide charge exchange atomic settings for the simulation")
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()

        solution = np.zeros([1,self.neq])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] == 0:
            return 0

        return calculate_ion_total_loss_cons(solution,self.atomic_parameters['iz'],
                                                    self.atomic_parameters['rec'],self.atomic_parameters['cx'],
                                                    self.parameters['adimensionalization']['temperature_scale'],
                                                    self.parameters['adimensionalization']['density_scale'],
                                                    self.parameters['physics']['Mref'],
                                                    self.parameters['physics']['R_E'],
                                                    self.parameters['adimensionalization']['charge_scale'],
                                                    self.parameters['adimensionalization']['mass_scale'],
                                                    self.parameters['adimensionalization']['speed_scale']**2*self.parameters['adimensionalization']['mass_scale'],
                                                    self.parameters['adimensionalization']['speed_scale'],
                                                    self._cons_idx)


    def Q_loss_tot(self,r,z):
        """
        returns value of total energy loss in given point (r,z)
        """
        if self.atomic_parameters is None:
            raise ValueError("Please, provide atomic settings for the simulation")
        if "iz" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide ionization atomic settings for the simulation")
        if "rec" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide recombination atomic settings for the simulation")
        if "cx" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide charge exchange atomic settings for the simulation")
        if "Eiz" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide Eiz atomic settings for the simulation")
        if "Erec" not in self.atomic_parameters.keys():
            raise ValueError("Please, provide Erec atomic settings for the simulation")
        if self._solution_interpolators is None:
            print('Definition of interpolators will take some time for the initialization')
            self.define_interpolators()

        solution = np.zeros([1,self.neq])
        for i in range(self.neq):
            solution[0,i] = self._solution_interpolators[i](r,z)
        if solution[0,0] == 0:
            return 0

        return calculate_total_loss_cons(solution,self.atomic_parameters['iz'],
                                                    self.atomic_parameters['rec'],self.atomic_parameters['cx'],
                                                    self.atomic_parameters['Eiz'],self.atomic_parameters['Erec'],
                                                    self.parameters['adimensionalization']['temperature_scale'],
                                                    self.parameters['adimensionalization']['density_scale'],
                                                    self.parameters['physics']['Mref'],
                                                    self.parameters['physics']['R_E'],
                                                    self.parameters['adimensionalization']['charge_scale'],
                                                    self.parameters['adimensionalization']['mass_scale'],
                                                    self.parameters['adimensionalization']['speed_scale']**2*self.parameters['adimensionalization']['mass_scale'],
                                                    self.parameters['adimensionalization']['speed_scale'],
                                                    self._cons_idx)
