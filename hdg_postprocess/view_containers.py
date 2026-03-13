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
class SolutionViewState:
    solution: SolutionFieldState = field(default_factory=SolutionFieldState)
    solution_skeleton: SolutionFieldState = field(default_factory=SolutionFieldState)
    gradient: SolutionFieldState = field(default_factory=SolutionFieldState)
    equilibrium: EquilibriumViewState = field(default_factory=EquilibriumViewState)
    derived: DerivedViewState = field(default_factory=DerivedViewState)


@dataclass
class SolutionViews:
    simple: SolutionViewState = field(default_factory=SolutionViewState)
    glob: SolutionViewState = field(default_factory=SolutionViewState)
    gauss: SolutionViewState = field(default_factory=SolutionViewState)
    boundary: SolutionViewState = field(default_factory=SolutionViewState)
    boundary_gauss: SolutionViewState = field(default_factory=SolutionViewState)
