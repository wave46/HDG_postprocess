import numpy as np


class SolutionFields:
    def __init__(self, solution):
        self._solution = solution

    def conservative(self, view="full", gradients=False):
        if view == "simple":
            if not self._solution.metadata.flags.combined_simple_solution:
                self._solution.recombine_simple_full_solution()
            return (
                self._solution.views.simple.gradient.conservative
                if gradients
                else self._solution.views.simple.solution.conservative
            )
        if view == "full":
            if not self._solution.metadata.flags.combined_to_full:
                self._solution.recombine_full_solution()
            return (
                self._solution.views.glob.gradient.conservative
                if gradients
                else self._solution.views.glob.solution.conservative
            )
        if view == "gauss":
            self._solution.calculate_in_gauss_points()
            return (
                self._solution.views.gauss.gradient.conservative
                if gradients
                else self._solution.views.gauss.solution.conservative
            )
        raise ValueError(f"Unsupported conservative view: {view}")

    def physical(self, view="full", gradients=False):
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
        raise ValueError(f"Unsupported physical view: {view}")


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
