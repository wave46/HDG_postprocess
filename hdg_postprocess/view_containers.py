from dataclasses import dataclass, field


@dataclass
class SolutionFieldState:
    conservative: object = None
    physical: object = None


@dataclass
class SolutionViewState:
    solution: SolutionFieldState = field(default_factory=SolutionFieldState)
    gradient: SolutionFieldState = field(default_factory=SolutionFieldState)


@dataclass
class SolutionViews:
    simple: SolutionViewState = field(default_factory=SolutionViewState)
    glob: SolutionViewState = field(default_factory=SolutionViewState)
