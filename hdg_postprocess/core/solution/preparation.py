import numpy as np


def ensure_simple_solution(solution):
    if not solution.metadata.flags.combined_simple_solution:
        print("Combining simple solution first")
        solution.assembly.simple()


def ensure_full_solution(solution):
    if not solution.metadata.flags.combined_to_full:
        print("Combining full solution first")
        solution.assembly.full()


def ensure_boundary_solution(solution):
    if not solution.metadata.flags.combined_boundary:
        print("Combining boundary solution first")
        solution.assembly.boundary()


def ensure_gauss_solution(solution):
    if not solution.metadata.flags.combined_gauss:
        print("Initializing values in gauss points first")
        solution.assembly.gauss()


def ensure_simple_physical(solution):
    if not solution.metadata.flags.simple_phys_initialized:
        print("Initializing physical solution first")
        solution.fields.initialize_physical("simple")


def ensure_full_physical(solution):
    if not solution.metadata.flags.full_phys_initialized:
        print("Initializing physical solution first")
        solution.fields.initialize_physical("full")


def ensure_connectivity_big(solution):
    if not solution.mesh.metadata.flags.connectivity_big_initialized:
        solution.mesh.geometry.connectivity_big


def ensure_gauss_volumes(solution):
    if not solution.mesh.metadata.flags.gauss_volumes_initialized:
        solution.mesh.geometry.gauss_volumes


def ensure_interpolators(solution):
    if solution.interpolators.solution is None:
        print("Definition of interpolators will take some time for the initialization")
        solution.sample.define_interpolators()


def project_full_to_simple(solution, full_values):
    n_vertices = solution.mesh.global_state.vertices.shape[0]
    trailing_shape = full_values.shape[2:]
    simple_values = np.zeros((n_vertices, *trailing_shape))
    simple_values[solution.mesh.global_state.connectivity.reshape(-1, 1).ravel()] = full_values.reshape(
        full_values.shape[0] * full_values.shape[1], *trailing_shape
    )
    return simple_values
