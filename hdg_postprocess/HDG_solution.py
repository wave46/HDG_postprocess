import numpy as np

from hdg_postprocess.api.solution import (
    AtomicRateState,
    InterpolatorState,
    SolutionMetadataState,
    ParameterState,
    RawPartitionState,
    SolutionSummaryState,
    SolutionViews,
    SolutionAssembly,
    SolutionAnalysis,
    SolutionEquilibrium,
    SolutionFields,
    SolutionNeutrals,
    SolutionTransport,
    SolutionPointwise,
    SolutionPlotting,
    SolutionSampling,
    SolutionSources,
    SolutionFluxSurface,
    SolutionTransport1D,
    SolutionTurbulence,
)


class HDGsolution:
    ""
    def __init__(self,raw_solutions, raw_solutions_skeleton, raw_gradients,
                 raw_equilibriums,raw_solution_boundary_infos, parameters,
                 n_partitions, mesh, raw_transport_1d=None, raw_neutral_flux_limiter_diagnostics=None):
        self._store_input_metadata(
            parameters,
            n_partitions,
            raw_equilibriums,
            raw_solution_boundary_infos,
            mesh,
            raw_transport_1d,
            raw_neutral_flux_limiter_diagnostics,
        )
        self._store_raw_partitions(raw_solutions, raw_solutions_skeleton, raw_gradients)
        self._neutral_flux_limiter_diagnostics = self._combine_neutral_flux_limiter_diagnostics()
        self._initial_setup()

    def _store_input_metadata(
        self,
        parameters,
        n_partitions,
        raw_equilibriums,
        raw_solution_boundary_infos,
        mesh,
        raw_transport_1d=None,
        raw_neutral_flux_limiter_diagnostics=None,
    ):
        self._parameters = parameters
        self._neq = parameters["Neq"][0]
        self._nphys = len(parameters["physics"]["physical_variable_names"])
        self._ndim = parameters["Ndim"][0]
        self._n_partitions = n_partitions
        self._mesh = mesh
        self._raw = RawPartitionState(
            solutions=[],
            solutions_skeleton=[],
            gradients=[],
            equilibriums=raw_equilibriums,
            boundary_infos=raw_solution_boundary_infos,
            transport_1d=raw_transport_1d if raw_transport_1d is not None else [],
            neutral_flux_limiter_diagnostics=(
                raw_neutral_flux_limiter_diagnostics
                if raw_neutral_flux_limiter_diagnostics is not None
                else []
            ),
        )

    def _store_raw_partitions(self, raw_solutions, raw_solutions_skeleton, raw_gradients):
        for raw_solution, raw_solution_skeleton, raw_gradient in zip(
            raw_solutions, raw_solutions_skeleton, raw_gradients
        ):
            self._raw.solutions.append(raw_solution.reshape(raw_solution.shape[0] // self.neq, self.neq))
            self._raw.solutions_skeleton.append(
                raw_solution_skeleton.reshape(raw_solution_skeleton.shape[0] // self.neq, self.neq)
            )
            raw_gradient = raw_gradient.reshape(raw_gradient.shape[0] // (self.neq * self.ndim), self.neq * self.ndim)
            self._raw.gradients.append(raw_gradient.reshape(raw_gradient.shape[0], self.neq, self.ndim))

    def _combine_neutral_flux_limiter_diagnostics(self):
        raw_diagnostics = self._raw.neutral_flux_limiter_diagnostics
        if not raw_diagnostics or not any(raw_diagnostics):
            return {}

        nodes_per_element = self.mesh.mesh_parameters["nodes_per_element"]
        keys = sorted({key for partition in raw_diagnostics for key in partition.keys()})
        combined = {}

        for key in keys:
            if self.n_partitions == 1:
                if key not in raw_diagnostics[0]:
                    continue
                combined[key] = self._reshape_neutral_flux_limiter_diagnostic(
                    raw_diagnostics[0][key],
                    self.mesh.raw.mesh_numbers[0]["Nelems"],
                    nodes_per_element,
                    key,
                )
                continue

            if not self.mesh.metadata.flags.combined_to_full:
                self.mesh.assembly.full()
            result = np.zeros((self.mesh.global_state.n_elements, nodes_per_element))
            filled = np.zeros(self.mesh.global_state.n_elements, dtype=bool)
            for i, partition in enumerate(raw_diagnostics):
                if key not in partition:
                    continue
                local_values = self._reshape_neutral_flux_limiter_diagnostic(
                    partition[key],
                    self.mesh.raw.mesh_numbers[i]["Nelems"],
                    nodes_per_element,
                    key,
                )
                mask = ~self.mesh.raw.ghost_elements[i].astype(bool).flatten()
                global_elements = self.mesh.raw.rest_mesh_data[i]["loc2glob_el"][mask]
                result[global_elements] = local_values[mask]
                filled[global_elements] = True
            if not np.all(filled):
                missing = np.flatnonzero(~filled)[:5]
                raise ValueError(
                    f"Neutral flux limiter diagnostic '{key}' is missing values for "
                    f"{np.count_nonzero(~filled)} global elements; first missing: {missing.tolist()}"
                )
            combined[key] = result
        return combined

    def _reshape_neutral_flux_limiter_diagnostic(self, values, n_elements, nodes_per_element, key):
        array = np.asarray(values)
        expected_size = n_elements * nodes_per_element
        if array.size != expected_size:
            raise ValueError(
                f"Neutral flux limiter diagnostic '{key}' has flat size {array.size}, "
                f"expected {expected_size} = {n_elements} * {nodes_per_element}."
            )
        return array.reshape(n_elements, nodes_per_element)

    def _initial_setup(self):
        self._init_state_containers()
        self._init_flags()
        self._init_variable_indices()
        self._init_charge_scale()
        self._normalize_external_heating_inputs()

    def _init_state_containers(self):
        self._views = SolutionViews()
        self._summary = SolutionSummaryState()
        self._metadata = SolutionMetadataState()
        self._parameter_state = ParameterState()
        self._atomic_rates = AtomicRateState()
        self._interpolators_state = InterpolatorState()
        self._assembly = SolutionAssembly(self)
        self._fields = SolutionFields(self)
        self._analysis = SolutionAnalysis(self)
        self._equilibrium = SolutionEquilibrium(self)
        self._sources = SolutionSources(self)
        self._neutrals = SolutionNeutrals(self)
        self._transport = SolutionTransport(self)
        self._turbulence = SolutionTurbulence(self)
        self._sample = SolutionSampling(self)
        self._plot = SolutionPlotting(self)
        self._pointwise = SolutionPointwise(self)
        self._flux_surface = SolutionFluxSurface(self)
        self._transport_1d = SolutionTransport1D(self)

    def _init_flags(self):
        self._metadata.flags.combined_simple_solution = False
        self._metadata.flags.full_phys_initialized = False
        self._metadata.flags.simple_phys_initialized = False
        self._metadata.flags.combined_to_full = False
        self._metadata.flags.combined_boundary = False
        self._metadata.flags.combined_gauss = False
        self._metadata.flags.combined_boundary_gauss = False
        self._metadata.flags.gauss_phys_initialized = False

    def _init_variable_indices(self):
        self._cons_idx = {}
        for i,label in enumerate(self.parameters['physics']['conservative_variable_names']):
            self._cons_idx[label] = i
        self._phys_idx = {}
        for i,label in enumerate(self.parameters['physics']['physical_variable_names']):
            self._phys_idx[label] = i
        self._metadata.indices.conservative = self._cons_idx
        self._metadata.indices.physical = self._phys_idx

    def _init_charge_scale(self):
        if 'charge_scale' in self.parameters['adimensionalization'].keys():
            self._e = self.parameters['adimensionalization']['charge_scale']
        else:
            self._e = 1.60217662e-19
        self._metadata.constants.elemental_charge = self._e
        if 'charge_scale' not in self.parameters['adimensionalization'].keys():
            self.parameters['adimensionalization']['charge_scale'] = self._e

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
    def metadata(self):
        """Public structured access to metadata, flags, and constants."""
        return self._metadata

    @property
    def additional_parameters(self):
        """Public structured access to derived setup parameters not stored in the simulation input."""
        return self._parameter_state

    @property
    def raw(self):
        """Public structured access to the raw partition payload."""
        return self._raw

    @property
    def neutral_flux_limiter_diagnostics(self):
        """Element-node neutral flux limiter diagnostics loaded from the solution file."""
        return self._neutral_flux_limiter_diagnostics

    @property
    def atomic_rates(self):
        """Public structured access to cached atomic rate coefficients."""
        return self._atomic_rates

    @property
    def interpolators(self):
        """Public structured access to cached interpolators."""
        return self._interpolators_state

    @property
    def fields(self):
        """Facade for conservative and physical field access."""
        return self._fields

    @property
    def assembly(self):
        """Facade for assembling and recombining view data."""
        return self._assembly

    @property
    def analysis(self):
        """Facade for power-balance and boundary summary workflows."""
        return self._analysis

    @property
    def equilibrium(self):
        """Facade for magnetic-axis and q-profile setup workflows."""
        return self._equilibrium

    @property
    def sources(self):
        """Facade for volumetric source and atomic-rate workflows."""
        return self._sources

    @property
    def neutrals(self):
        """Facade for neutral transport-derived workflows."""
        return self._neutrals

    @property
    def transport(self):
        """Facade for transport-model postprocessing workflows."""
        return self._transport

    @property
    def turbulence(self):
        """Facade for turbulence-model-derived workflows."""
        return self._turbulence

    @property
    def sample(self):
        """Facade for point and line sampling workflows."""
        return self._sample

    @property
    def plot(self):
        """Facade for high-level plotting workflows."""
        return self._plot

    @property
    def pointwise(self):
        """Facade for direct pointwise sampling of derived quantities."""
        return self._pointwise

    @property
    def flux_surface(self):
        """Facade for flux-surface-reduced postprocessing quantities."""
        return self._flux_surface


    @property
    def transport_1d(self):
        """Facade for optional 1D transport profiles stored in the solution file."""
        return self._transport_1d
