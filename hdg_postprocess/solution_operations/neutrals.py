import numpy as np

from hdg_postprocess.routines.neutrals import (
    calculate_dnn_cons,
    calculate_dnn_with_nn_collision_cons,
    calculate_mfp_cons,
)


def calculate_dnn(solution, which="simple"):
    _ensure_neutral_settings(solution)
    if which == "simple":
        _ensure_simple_phys(solution)
        solution.calculate_dnn(which="full")
        _assign_simple_view(solution, "_dnn_simple", solution._dnn)
    if which == "full":
        _ensure_full_solution(solution)
        solution._dnn = calculate_dnn_cons(
            solution.solution_glob,
            solution.dnn_parameters,
            solution.atomic_parameters,
            solution._e,
            solution.parameters["adimensionalization"]["mass_scale"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["length_scale"],
            solution.parameters["adimensionalization"]["time_scale"],
        )


def calculate_dnn_with_nn_collision(solution, which="simple"):
    _ensure_neutral_settings(solution)
    if which == "simple":
        _ensure_simple_phys(solution)
        solution.calculate_dnn_with_nn_collision(which="full")
        _assign_simple_view(solution, "_dnn_with_nn_collision_simple", solution._dnn_with_nn_collision)
        # Preserve legacy attribute spellings used by old properties.
        solution._dnn_simple_with_nn_collision = solution._dnn_with_nn_collision
        solution._dnn_simple_with_nn_collision_simple = solution._dnn_with_nn_collision_simple
    if which == "full":
        _ensure_full_solution(solution)
        solution._dnn_with_nn_collision = calculate_dnn_with_nn_collision_cons(
            solution.solution_glob,
            solution.dnn_parameters,
            solution.atomic_parameters,
            solution._e,
            solution.parameters["adimensionalization"]["mass_scale"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["length_scale"],
            solution.parameters["adimensionalization"]["time_scale"],
        )
        solution._dnn_simple_with_nn_collision = solution._dnn_with_nn_collision


def calculate_mfp(solution, which="simple"):
    _ensure_neutral_settings(solution)
    if not solution._simple_phys_initialized:
        print("Initializing physical solution first")
        solution.init_phys_variables("both")
    if which == "simple":
        _ensure_simple_phys(solution)
        solution.calculate_mfp(which="full")
        _assign_simple_view(solution, "_mfp_simple", solution._mfp)
    if which == "full":
        _ensure_full_solution(solution)
        if not solution._simple_phys_initialized:
            print("Initializing physical solution first")
            solution.init_phys_variables("full")
        if solution._dnn is None:
            solution.calculate_dnn("full")
        solution._mfp = calculate_mfp_cons(
            solution.solution_glob,
            solution.dnn_parameters,
            solution.atomic_parameters,
            solution._e,
            solution.parameters["adimensionalization"]["mass_scale"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["length_scale"],
            solution.parameters["adimensionalization"]["time_scale"],
        )


def _ensure_neutral_settings(solution):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if solution.dnn_parameters is None:
        raise ValueError("Please, provide neutral diffusion settings for the simulation")
    if "iz" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide ionization atomic settings for the simulation")
    if "cx" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide ionization atomic settings for the simulation")


def _ensure_simple_phys(solution):
    if not solution._simple_phys_initialized:
        print("Initializing physical solution first")
        solution.init_phys_variables("simple")


def _ensure_full_solution(solution):
    if not solution._combined_to_full:
        solution.recombine_full_solution()


def _assign_simple_view(solution, attribute_name, full_values):
    simple_values = np.zeros(solution.mesh.vertices_glob.shape[0])
    simple_values[solution.mesh.connectivity_glob.reshape(-1, 1).ravel()] = full_values.reshape(
        solution.solution_glob.shape[0] * solution.solution_glob.shape[1]
    )
    setattr(solution, attribute_name, simple_values)
