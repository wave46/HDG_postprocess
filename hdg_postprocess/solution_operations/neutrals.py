import numpy as np

from hdg_postprocess.routines.neutrals import (
    calculate_dnn_cons,
    calculate_dnn_with_nn_collision_cons,
    calculate_mfp_cons,
)
from hdg_postprocess.solution_operations import physical as physical_ops


def calculate_dnn(solution, which="simple"):
    _ensure_neutral_settings(solution)
    if which == "simple":
        _ensure_simple_phys(solution)
        calculate_dnn(solution, which="full")
        solution.views.simple.derived.dnn = _simple_values(solution, solution.views.glob.derived.dnn)
    if which == "full":
        _ensure_full_solution(solution)
        solution.views.glob.derived.dnn = calculate_dnn_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.neutral_diffusion,
            solution.additional_parameters.atomic,
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
        calculate_dnn_with_nn_collision(solution, which="full")
        solution.views.simple.derived.dnn_with_nn_collision = _simple_values(
            solution, solution.views.glob.derived.dnn_with_nn_collision
        )
    if which == "full":
        _ensure_full_solution(solution)
        solution.views.glob.derived.dnn_with_nn_collision = calculate_dnn_with_nn_collision_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.neutral_diffusion,
            solution.additional_parameters.atomic,
            solution._e,
            solution.parameters["adimensionalization"]["mass_scale"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["length_scale"],
            solution.parameters["adimensionalization"]["time_scale"],
        )
def calculate_mfp(solution, which="simple"):
    _ensure_neutral_settings(solution)
    if not solution.metadata.flags.simple_phys_initialized:
        print("Initializing physical solution first")
        physical_ops.init_phys_variables(solution, "both")
    if which == "simple":
        _ensure_simple_phys(solution)
        calculate_mfp(solution, which="full")
        solution.views.simple.derived.mfp = _simple_values(solution, solution.views.glob.derived.mfp)
    if which == "full":
        _ensure_full_solution(solution)
        if not solution.metadata.flags.simple_phys_initialized:
            print("Initializing physical solution first")
            physical_ops.init_phys_variables(solution, "full")
        if solution.views.glob.derived.dnn is None:
            calculate_dnn(solution, "full")
        solution.views.glob.derived.mfp = calculate_mfp_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.neutral_diffusion,
            solution.additional_parameters.atomic,
            solution._e,
            solution.parameters["adimensionalization"]["mass_scale"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["length_scale"],
            solution.parameters["adimensionalization"]["time_scale"],
        )


def _ensure_neutral_settings(solution):
    if solution.additional_parameters.atomic is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if solution.additional_parameters.neutral_diffusion is None:
        raise ValueError("Please, provide neutral diffusion settings for the simulation")
    if "iz" not in solution.additional_parameters.atomic.keys():
        raise ValueError("Please, provide ionization atomic settings for the simulation")
    if "cx" not in solution.additional_parameters.atomic.keys():
        raise ValueError("Please, provide ionization atomic settings for the simulation")


def _ensure_simple_phys(solution):
    if not solution.metadata.flags.simple_phys_initialized:
        print("Initializing physical solution first")
        physical_ops.init_phys_variables(solution, "simple")


def _ensure_full_solution(solution):
    if not solution.metadata.flags.combined_to_full:
        solution.assembly.full()


def _simple_values(solution, full_values):
    simple_values = np.zeros(solution.mesh.vertices_glob.shape[0])
    simple_values[solution.mesh.connectivity_glob.reshape(-1, 1).ravel()] = full_values.reshape(
        solution.views.glob.solution.conservative.shape[0] * solution.views.glob.solution.conservative.shape[1]
    )
    return simple_values
