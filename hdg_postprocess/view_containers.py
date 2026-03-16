from dataclasses import dataclass, field


@dataclass
class SolutionFieldState:
    conservative: object = None
    physical: object = None


@dataclass
class EquilibriumViewState:
    magnetic_field: object = None
    magnetic_field_unit: object = None
    jtor: object = None
    poloidal_flux: object = None
    a: object = None
    qcyl: object = None


@dataclass
class DerivedViewState:
    dnn: object = None
    dnn_with_nn_collision: object = None
    mfp: object = None
    dk: object = None


@dataclass
class SourceViewState:
    ionization_source: object = None
    ion_gain_iz: object = None
    ion_sink_rec: object = None
    ion_sink_cx: object = None
    electron_sink_iz: object = None
    electron_sink_rec: object = None
    electron_gain_rec: object = None
    electron_sink_cooling_factor: object = None
    cooling_factor: object = None
    cx_source: object = None
    external_heating: object = None
    external_heating_e: object = None
    external_heating_i: object = None
    ohmic_source: object = None


@dataclass
class SourceSummaryState:
    ion_gain_iz_total: object = None
    ion_sink_rec_total: object = None
    ion_sink_cx_total: object = None
    electron_sink_iz_total: object = None
    electron_sink_rec_total: object = None
    electron_gain_rec_total: object = None
    electron_sink_cooling_factor_total: object = None
    external_heating_total: object = None
    external_heating_e_total: object = None
    external_heating_i_total: object = None
    ohmic_source_total: object = None


@dataclass
class BoundarySummaryState:
    profile: object = None
    ion_energy_sheath_loss_total: object = None
    electron_energy_sheath_loss_total: object = None


@dataclass
class AxisState:
    r: object = None
    z: object = None


@dataclass
class EquilibriumSummaryState:
    axis: AxisState = field(default_factory=AxisState)


@dataclass
class SolutionSummaryState:
    sources: SourceSummaryState = field(default_factory=SourceSummaryState)
    boundary: BoundarySummaryState = field(default_factory=BoundarySummaryState)
    equilibrium: EquilibriumSummaryState = field(default_factory=EquilibriumSummaryState)


@dataclass
class SolutionFlagState:
    combined_to_full: bool = False
    combined_boundary: bool = False
    combined_simple_solution: bool = False
    combined_gauss: bool = False
    combined_boundary_gauss: bool = False
    full_phys_initialized: bool = False
    simple_phys_initialized: bool = False
    gauss_phys_initialized: bool = False


@dataclass
class SolutionIndexState:
    conservative: object = None
    physical: object = None


@dataclass
class PhysicalConstantState:
    elemental_charge: object = None


@dataclass
class SolutionCacheState:
    boundary_gauss_boundaries: object = None
    boundary_gauss_ordering: object = None
    boundary_gauss_connectivity: object = None
    boundary_gauss_face_elements: object = None


@dataclass
class SolutionMetadataState:
    flags: SolutionFlagState = field(default_factory=SolutionFlagState)
    indices: SolutionIndexState = field(default_factory=SolutionIndexState)
    constants: PhysicalConstantState = field(default_factory=PhysicalConstantState)
    cache: SolutionCacheState = field(default_factory=SolutionCacheState)


@dataclass
class ParameterState:
    atomic: object = None
    neutral_diffusion: object = None
    turbulence: object = None

    def set_atomic(self, value):
        self.atomic = value

    def set_neutral_diffusion(self, value, adimensionalization):
        self.neutral_diffusion = value
        if value is None:
            return
        self.neutral_diffusion["dnn_max_adim"] = (
            self.neutral_diffusion["dnn_max"]
            / adimensionalization["length_scale"] ** 2
            * adimensionalization["time_scale"]
        )
        if not value["const"]:
            self.neutral_diffusion["dnn_min_adim"] = (
                self.neutral_diffusion["dnn_min"]
                / adimensionalization["length_scale"] ** 2
                * adimensionalization["time_scale"]
            )

    def set_turbulence(self, value, adimensionalization):
        self.turbulence = value
        if value is None:
            return
        self.turbulence["dk_max_adim"] = (
            self.turbulence["dk_max"]
            / adimensionalization["length_scale"] ** 2
            * adimensionalization["time_scale"]
        )
        self.turbulence["dk_min_adim"] = (
            self.turbulence["dk_min"]
            / adimensionalization["length_scale"] ** 2
            * adimensionalization["time_scale"]
        )


@dataclass
class AtomicRateState:
    ionization_simple: object = None
    recombination_simple: object = None
    cx_simple: object = None


@dataclass
class RawPartitionState:
    solutions: object = None
    solutions_skeleton: object = None
    gradients: object = None
    equilibriums: object = None
    boundary_infos: object = None


@dataclass
class InterpolatorState:
    sample: object = None
    solution: object = None
    gradient: object = None
    field: object = None
    qcyl: object = None


@dataclass
class SolutionViewState:
    solution: SolutionFieldState = field(default_factory=SolutionFieldState)
    solution_skeleton: SolutionFieldState = field(default_factory=SolutionFieldState)
    gradient: SolutionFieldState = field(default_factory=SolutionFieldState)
    equilibrium: EquilibriumViewState = field(default_factory=EquilibriumViewState)
    derived: DerivedViewState = field(default_factory=DerivedViewState)
    sources: SourceViewState = field(default_factory=SourceViewState)


@dataclass
class SolutionViews:
    simple: SolutionViewState = field(default_factory=SolutionViewState)
    glob: SolutionViewState = field(default_factory=SolutionViewState)
    gauss: SolutionViewState = field(default_factory=SolutionViewState)
    boundary: SolutionViewState = field(default_factory=SolutionViewState)
    boundary_gauss: SolutionViewState = field(default_factory=SolutionViewState)
