import numpy as np

from hdg_postprocess.solution_operations import assembly as assembly_ops
from hdg_postprocess.solution_operations import magnetic_equilibrium as equilibrium_ops
from hdg_postprocess.solution_operations import neutrals as neutrals_ops
from hdg_postprocess.solution_operations import plasma_sources as plasma_source_ops
from hdg_postprocess.solution_operations import turbulent_model as turbulent_model_ops


class SolutionFields:
    def __init__(self, solution):
        self._solution = solution

    def conservative(self, view="full", gradients=False, skeleton=False, boundaries=None):
        if view == "simple":
            if not self._solution.metadata.flags.combined_simple_solution:
                self._solution.assembly.simple()
            return (
                self._solution.views.simple.gradient.conservative
                if gradients
                else self._solution.views.simple.solution.conservative
            )
        if view == "full":
            if not self._solution.metadata.flags.combined_to_full:
                self._solution.assembly.full()
            return (
                self._solution.views.glob.gradient.conservative
                if gradients
                else self._solution.views.glob.solution.conservative
            )
        if view == "gauss":
            if not self._solution.metadata.flags.combined_gauss:
                self._solution.assembly.gauss()
            return (
                self._solution.views.gauss.gradient.conservative
                if gradients
                else self._solution.views.gauss.solution.conservative
            )
        if view == "boundary":
            if not self._solution.metadata.flags.combined_boundary:
                self._solution.assembly.boundary()
            if gradients and skeleton:
                raise ValueError("Boundary gradient and solution skeleton are distinct views.")
            if gradients:
                return self._solution.views.boundary.gradient.conservative
            if skeleton:
                return self._solution.views.boundary.solution_skeleton.conservative
            return self._solution.views.boundary.solution.conservative
        if view == "boundary_gauss":
            if not self._solution.metadata.flags.combined_boundary_gauss:
                if boundaries is None:
                    boundaries = np.unique(self._solution.raw.boundary_infos[0]["boundary_flags"])
                self._solution.assembly.boundary_gauss(boundaries)
            if gradients and skeleton:
                raise ValueError("Boundary gradient and solution skeleton are distinct views.")
            if gradients:
                return self._solution.views.boundary_gauss.gradient.conservative
            if skeleton:
                return self._solution.views.boundary_gauss.solution_skeleton.conservative
            return self._solution.views.boundary_gauss.solution.conservative
        raise ValueError(f"Unsupported conservative view: {view}")

    def physical(self, view="full", gradients=False, skeleton=False):
        if view == "simple":
            if not self._solution.metadata.flags.simple_phys_initialized:
                self._solution.init_phys_variables("simple")
            return (
                self._solution.views.simple.gradient.physical
                if gradients
                else self._solution.views.simple.solution.physical
            )
        if view == "full":
            if not self._solution.metadata.flags.full_phys_initialized:
                self._solution.init_phys_variables("full")
            return (
                self._solution.views.glob.gradient.physical
                if gradients
                else self._solution.views.glob.solution.physical
            )
        if view == "gauss":
            if not self._solution.metadata.flags.gauss_phys_initialized:
                self._solution.init_phys_variables("gauss")
            return (
                self._solution.views.gauss.gradient.physical
                if gradients
                else self._solution.views.gauss.solution.physical
            )
        if view in {"boundary", "boundary_gauss"}:
            raise ValueError("Physical boundary fields are not initialized yet; use conservative boundary data for now.")
        raise ValueError(f"Unsupported physical view: {view}")


class SolutionAssembly:
    def __init__(self, solution):
        self._solution = solution

    def full(self):
        assembly_ops.recombine_full_solution(self._solution)
        return self._solution.views.glob

    def simple(self):
        assembly_ops.recombine_simple_full_solution(self._solution)
        return self._solution.views.simple

    def boundary(self):
        assembly_ops.recombine_boundary_solution(self._solution)
        return self._solution.views.boundary

    def gauss(self):
        assembly_ops.calculate_in_gauss_points(self._solution)
        return self._solution.views.gauss

    def boundary_gauss(self, boundaries=None):
        if boundaries is None:
            boundaries = np.unique(self._solution.raw.boundary_infos[0]["boundary_flags"])
        assembly_ops.calculate_in_boundary_gauss_points(self._solution, boundaries)
        return self._solution.views.boundary_gauss


class SolutionEquilibrium:
    def __init__(self, solution):
        self._solution = solution

    def define_axis(self):
        equilibrium_ops.define_magnetic_axis(self._solution)
        return self._solution.summary.equilibrium.axis

    def define_minor_radii(self, view="simple"):
        which = "full" if view == "glob" else view
        equilibrium_ops.define_minor_radii(self._solution, which=which)
        target_view = self._solution.views.glob if which == "full" else getattr(self._solution.views, which)
        return target_view.equilibrium.a

    def define_qcyl(self, view="simple"):
        which = "full" if view == "glob" else view
        equilibrium_ops.define_qcyl(self._solution, which=which)
        target_view = self._solution.views.glob if which == "full" else getattr(self._solution.views, which)
        return target_view.equilibrium.qcyl


class SolutionSources:
    def __init__(self, solution):
        self._solution = solution

    def ohmic(self, view="simple"):
        plasma_source_ops.calculate_ohmic_source(self._solution, view)
        target_view = self._solution.views.glob if view == "full" else getattr(self._solution.views, view)
        return target_view.sources.ohmic_source

    def ionization_rate(self, view="simple"):
        plasma_source_ops.calculate_ionization_rate(self._solution, view)
        return self._solution.atomic_rates.ionization_simple

    def recombination_rate(self, view="simple"):
        plasma_source_ops.calculate_recombination_rate(self._solution, view)
        return self._solution.atomic_rates.recombination_simple

    def cx_rate(self, view="simple"):
        plasma_source_ops.calculate_cx_rate(self._solution, view)
        return self._solution.atomic_rates.cx_simple

    def ionization(self, view="simple"):
        plasma_source_ops.calculate_ionization_source(self._solution, view)
        target_view = self._solution.views.glob if view == "full" else getattr(self._solution.views, view)
        return target_view.sources.ionization_source

    def ion_gain_iz(self, view="simple"):
        plasma_source_ops.calculate_ion_gain_due_to_iz(self._solution, view)
        target_view = self._solution.views.glob if view == "full" else getattr(self._solution.views, view)
        return target_view.sources.ion_gain_iz

    def electron_sink_iz(self, view="simple"):
        plasma_source_ops.calculate_electron_sink_due_to_iz(self._solution, view)
        target_view = self._solution.views.glob if view == "full" else getattr(self._solution.views, view)
        return target_view.sources.electron_sink_iz

    def electron_sink_rec(self, view="simple"):
        plasma_source_ops.calculate_electron_sink_due_to_rec(self._solution, view)
        target_view = self._solution.views.glob if view == "full" else getattr(self._solution.views, view)
        return target_view.sources.electron_sink_rec

    def electron_gain_rec(self, view="simple"):
        plasma_source_ops.calculate_electron_gain_due_to_rec(self._solution, view)
        target_view = self._solution.views.glob if view == "full" else getattr(self._solution.views, view)
        return target_view.sources.electron_gain_rec

    def ion_sink_rec(self, view="simple"):
        plasma_source_ops.calculate_ion_sink_due_to_rec(self._solution, view)
        target_view = self._solution.views.glob if view == "full" else getattr(self._solution.views, view)
        return target_view.sources.ion_sink_rec

    def ion_sink_cx(self, view="simple"):
        plasma_source_ops.calculate_ion_sink_due_to_cx(self._solution, view)
        target_view = self._solution.views.glob if view == "full" else getattr(self._solution.views, view)
        return target_view.sources.ion_sink_cx

    def electron_sink_cooling_factor(self, view="simple"):
        plasma_source_ops.calculate_electron_sink_due_to_cooling_factor(self._solution, view)
        target_view = self._solution.views.glob if view == "full" else getattr(self._solution.views, view)
        return target_view.sources.electron_sink_cooling_factor

    def cooling_factor(self, view="simple"):
        plasma_source_ops.calculate_cooling_factor(self._solution, view)
        target_view = self._solution.views.glob if view == "full" else getattr(self._solution.views, view)
        return target_view.sources.cooling_factor

    def cx(self, view="simple"):
        plasma_source_ops.calculate_cx_source(self._solution, view)
        target_view = self._solution.views.glob if view == "full" else getattr(self._solution.views, view)
        return target_view.sources.cx_source


class SolutionNeutrals:
    def __init__(self, solution):
        self._solution = solution

    def dnn(self, view="simple", with_nn_collision=False):
        if with_nn_collision:
            neutrals_ops.calculate_dnn_with_nn_collision(self._solution, view)
            if view == "full":
                return self._solution.views.glob.derived.dnn_with_nn_collision
            return self._solution.views.simple.derived.dnn_with_nn_collision
        neutrals_ops.calculate_dnn(self._solution, view)
        if view == "full":
            return self._solution.views.glob.derived.dnn
        return self._solution.views.simple.derived.dnn

    def mfp(self, view="simple"):
        neutrals_ops.calculate_mfp(self._solution, view)
        if view == "full":
            return self._solution.views.glob.derived.mfp
        return self._solution.views.simple.derived.mfp


class SolutionTurbulence:
    def __init__(self, solution):
        self._solution = solution

    def dk(self, view="simple"):
        turbulent_model_ops.calculate_dk(self._solution, view)
        if view == "full":
            return self._solution.views.glob.derived.dk
        return self._solution.views.simple.derived.dk


class SolutionAnalysis:
    def __init__(self, solution):
        self._solution = solution

    def power_balance(self):
        return self._solution.calculate_power_balance()

    def boundary_summary(self):
        self._solution.calculate_boundary_summary()
        return self._solution.summary.boundary.profile


class SolutionSampling:
    def __init__(self, solution):
        self._solution = solution

    def point(self, r, z, variables):
        variable_list = [variables] if isinstance(variables, str) else list(variables)
        sampled = self.line(np.asarray([r]), np.asarray([z]), variable_list)
        if isinstance(variables, str):
            return sampled[variables][0]
        return {name: values[0] for name, values in sampled.items()}

    def line(self, r_line, z_line, variables):
        self._prepare_for_sampling()
        variable_list = [variables] if isinstance(variables, str) else list(variables)
        return self._solution.calculate_variables_along_line(np.asarray(r_line), np.asarray(z_line), variable_list)

    def _prepare_for_sampling(self):
        if not self._solution.metadata.flags.simple_phys_initialized:
            self._solution.init_phys_variables("simple")
        if self._solution.interpolators.solution is None:
            self._solution.define_interpolators()


class SolutionPlotting:
    def __init__(self, solution):
        self._solution = solution

    def overview(self, *args, **kwargs):
        return self._solution.plot_overview(*args, **kwargs)

    def physical_overview(self, *args, **kwargs):
        if not self._solution.metadata.flags.simple_phys_initialized:
            self._solution.init_phys_variables("simple")
        return self._solution.plot_overview_physical(*args, **kwargs)
