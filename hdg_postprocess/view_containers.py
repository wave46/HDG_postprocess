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
