import numpy as np

from hdg_postprocess.core.solution import analysis as analysis_ops
from hdg_postprocess.core.solution import assembly as assembly_ops
from hdg_postprocess.core.solution import boundary as boundary_ops
from hdg_postprocess.core.solution import magnetic_equilibrium as equilibrium_ops
from hdg_postprocess.core.solution import physical as physical_ops
from hdg_postprocess.core.solution import plotting as plotting_ops
from hdg_postprocess.core.solution import sampling as sampling_ops


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
                self.initialize_physical("simple")
            return (
                self._solution.views.simple.gradient.physical
                if gradients
                else self._solution.views.simple.solution.physical
            )
        if view == "full":
            if not self._solution.metadata.flags.full_phys_initialized:
                self.initialize_physical("full")
            return (
                self._solution.views.glob.gradient.physical
                if gradients
                else self._solution.views.glob.solution.physical
            )
        if view == "gauss":
            if not self._solution.metadata.flags.gauss_phys_initialized:
                self.initialize_physical("gauss")
            return (
                self._solution.views.gauss.gradient.physical
                if gradients
                else self._solution.views.gauss.solution.physical
            )
        if view in {"boundary", "boundary_gauss"}:
            raise ValueError("Physical boundary fields are not initialized yet; use conservative boundary data for now.")
        raise ValueError(f"Unsupported physical view: {view}")

    def initialize_physical(self, view="both"):
        physical_ops.init_phys_variables(self._solution, which=view)

    def to_physical(self, data):
        return physical_ops.cons2phys(self._solution, data)


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

    def magnetic_field(self, view="simple"):
        target_view = self._prepare_view(view)
        return target_view.equilibrium.magnetic_field

    def poloidal_flux(self, view="simple"):
        target_view = self._prepare_view(view)
        return target_view.equilibrium.poloidal_flux

    def jtor(self, view="simple"):
        target_view = self._prepare_view(view)
        if view in {"boundary", "boundary_gauss"}:
            raise ValueError("jtor is not exposed on boundary equilibrium views.")
        return target_view.equilibrium.jtor

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

    def _prepare_view(self, view):
        normalized = "full" if view == "glob" else view
        if normalized == "full":
            if not self._solution.metadata.flags.combined_to_full:
                self._solution.assembly.full()
            return self._solution.views.glob
        if normalized == "simple":
            if not self._solution.metadata.flags.combined_simple_solution:
                self._solution.assembly.simple()
            return self._solution.views.simple
        if normalized == "gauss":
            if not self._solution.metadata.flags.combined_gauss:
                self._solution.assembly.gauss()
            return self._solution.views.gauss
        if normalized == "boundary":
            if not self._solution.metadata.flags.combined_boundary:
                self._solution.assembly.boundary()
            return self._solution.views.boundary
        if normalized == "boundary_gauss":
            if not self._solution.metadata.flags.combined_boundary_gauss:
                self._solution.assembly.boundary_gauss()
            return self._solution.views.boundary_gauss
        raise ValueError(f"Unsupported equilibrium view: {view}")


class SolutionAnalysis:
    def __init__(self, solution):
        self._solution = solution

    def power_balance(self):
        return analysis_ops.calculate_power_balance(self._solution)

    def volumetric_sources(self):
        return analysis_ops.calculate_volumetric_sources(self._solution)

    def power_losses_to_wall(self):
        return analysis_ops.calculate_power_losses_to_wall(self._solution)

    def boundary_summary(self):
        boundary_ops.calculate_boundary_summary(self._solution)
        return self._solution.summary.boundary.profile

    def wall_profile(self):
        return boundary_ops.summary_along_the_wall(self._solution)


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
        return sampling_ops.calculate_variables_along_line(self._solution, np.asarray(r_line), np.asarray(z_line), variable_list)

    def save_line(self, save_folder, r_line, z_line, variables):
        self._prepare_for_sampling()
        variable_list = [variables] if isinstance(variables, str) else list(variables)
        return sampling_ops.save_summary_line(self._solution, save_folder, np.asarray(r_line), np.asarray(z_line), variable_list)

    def define_interpolators(self):
        return sampling_ops.define_interpolators(self._solution)

    def _prepare_for_sampling(self):
        if not self._solution.metadata.flags.simple_phys_initialized:
            physical_ops.init_phys_variables(self._solution, which="simple")
        if self._solution.interpolators.solution is None:
            sampling_ops.define_interpolators(self._solution)


class SolutionPlotting:
    def __init__(self, solution):
        self._solution = solution

    def overview(self, *args, **kwargs):
        return plotting_ops.plot_overview(self._solution, *args, **kwargs)

    def overview_difference(self, *args, **kwargs):
        return plotting_ops.plot_overview_difference(self._solution, *args, **kwargs)

    def physical_overview(self, *args, **kwargs):
        if not self._solution.metadata.flags.simple_phys_initialized:
            physical_ops.init_phys_variables(self._solution, which="simple")
        return plotting_ops.plot_overview_physical(self._solution, *args, **kwargs)

    def physical_overview_difference(self, *args, **kwargs):
        if not self._solution.metadata.flags.simple_phys_initialized:
            physical_ops.init_phys_variables(self._solution, which="simple")
        return plotting_ops.plot_overview_physical_difference(self._solution, *args, **kwargs)

    def variables_overview(self, *args, **kwargs):
        return plotting_ops.plot_variables_overview(self._solution, *args, **kwargs)
