import numpy as np

from hdg_postprocess.formats import load_HDG_mesh_from_file, load_HDG_solution_from_file


def load_solution(solpath, solname_base, meshpath=None, meshname_base=None, n_partitions=1):
    """Load a solution through the modern additive API."""
    legacy = load_HDG_solution_from_file(solpath, solname_base, meshpath, meshname_base, n_partitions)
    return PostprocessedSolution(legacy)


def load_mesh(meshpath, meshname_base, n_partitions=1):
    """Load a mesh through the modern additive API."""
    legacy = load_HDG_mesh_from_file(meshpath, meshname_base, n_partitions)
    return PostprocessedMesh(legacy)


class PostprocessedSolution:
    def __init__(self, legacy_solution):
        self.legacy = legacy_solution
        self.fields = _SolutionFields(legacy_solution)
        self.analysis = _SolutionAnalysis(legacy_solution)
        self.sample = _SolutionSampling(legacy_solution)
        self.plot = _SolutionPlotting(legacy_solution)

    @property
    def mesh(self):
        return PostprocessedMesh(self.legacy.mesh)

    @property
    def views(self):
        return self.legacy.views

    @property
    def summary(self):
        return self.legacy.summary

    @property
    def parameters_state(self):
        return self.legacy.parameter_state

    @property
    def atomic_rates(self):
        return self.legacy.atomic_rates

    @property
    def interpolators(self):
        return self.legacy.interpolators

    @property
    def metadata(self):
        return {
            "neq": self.legacy.neq,
            "nphys": self.legacy.nphys,
            "n_partitions": self.legacy.n_partitions,
            "parameters": self.legacy.parameters,
            "cons_idx": self.legacy.cons_idx,
            "phys_idx": self.legacy.phys_idx,
        }


class PostprocessedMesh:
    def __init__(self, legacy_mesh):
        self.legacy = legacy_mesh
        self.plot = _MeshPlotting(legacy_mesh)
        self.topology = _MeshTopology(legacy_mesh)

    @property
    def metadata(self):
        return {
            "n_partitions": self.legacy.n_partitions,
            "mesh_parameters": self.legacy.mesh_parameters,
            "mesh_extent": self.legacy.mesh_extent,
            "p_order": self.legacy.p_order,
        }


class _SolutionFields:
    def __init__(self, solution):
        self._solution = solution

    def conservative(self, view="full", gradients=False):
        if view == "simple":
            if not self._solution.combined_simple_solution:
                self._solution.recombine_simple_full_solution()
            return self._solution.views.simple.gradient.conservative if gradients else self._solution.views.simple.solution.conservative
        if view == "full":
            if not self._solution.combined_to_full:
                self._solution.recombine_full_solution()
            return self._solution.views.glob.gradient.conservative if gradients else self._solution.views.glob.solution.conservative
        if view == "gauss":
            self._solution.calculate_in_gauss_points()
            return self._solution.views.gauss.gradient.conservative if gradients else self._solution.views.gauss.solution.conservative
        raise ValueError(f"Unsupported conservative view: {view}")

    def physical(self, view="full", gradients=False):
        if view == "simple":
            if not self._solution.simple_phys_initialized:
                self._solution.init_phys_variables("simple")
            return self._solution.views.simple.gradient.physical if gradients else self._solution.views.simple.solution.physical
        if view == "full":
            if not self._solution.full_phys_initialized:
                self._solution.init_phys_variables("full")
            return self._solution.views.glob.gradient.physical if gradients else self._solution.views.glob.solution.physical
        raise ValueError(f"Unsupported physical view: {view}")


class _SolutionAnalysis:
    def __init__(self, solution):
        self._solution = solution

    def power_balance(self):
        return self._solution.calculate_power_balance()

    def boundary_summary(self):
        self._solution.calculate_boundary_summary()
        return self._solution.summary.boundary.boundary_summary


class _SolutionSampling:
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
        if not self._solution.simple_phys_initialized:
            self._solution.init_phys_variables("simple")
        if self._solution.solution_interpolators is None:
            self._solution.define_interpolators()


class _SolutionPlotting:
    def __init__(self, solution):
        self._solution = solution

    def overview(self, *args, **kwargs):
        return self._solution.plot_overview(*args, **kwargs)

    def physical_overview(self, *args, **kwargs):
        if not self._solution.simple_phys_initialized:
            self._solution.init_phys_variables("simple")
        return self._solution.plot_overview_physical(*args, **kwargs)


class _MeshPlotting:
    def __init__(self, mesh):
        self._mesh = mesh

    def raw(self, *args, **kwargs):
        return self._mesh.plot_raw_meshes(*args, **kwargs)

    def full(self, *args, **kwargs):
        if not self._mesh.combined_to_full:
            self._mesh.recombine_full_mesh()
        return self._mesh.plot_full_mesh(*args, **kwargs)


class _MeshTopology:
    def __init__(self, mesh):
        self._mesh = mesh

    def recombine_full(self):
        self._mesh.recombine_full_mesh()
        return self._mesh

    def create_big_connectivity(self):
        self._mesh.create_connectivity_big()
        return self._mesh.connectivity_big
