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
class ParameterState:
    atomic: object = None
    neutral_diffusion: object = None
    turbulence: object = None


@dataclass
class AtomicRateState:
    ionization_simple: object = None
    recombination_simple: object = None
    cx_simple: object = None


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
