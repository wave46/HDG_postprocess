from hdg_postprocess.routines.neutrals import (
    calculate_dnn_cons,
    calculate_dnn_with_nn_collision_cons,
    calculate_mfp_cons,
)
from hdg_postprocess.solution_operations import preparation as prep_ops


def calculate_dnn(solution, which="simple"):
    _ensure_neutral_settings(solution)
    if which == "simple":
        prep_ops.ensure_simple_physical(solution)
        calculate_dnn(solution, which="full")
        solution.views.simple.derived.dnn = prep_ops.project_full_to_simple(solution, solution.views.glob.derived.dnn)
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
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
        prep_ops.ensure_simple_physical(solution)
        calculate_dnn_with_nn_collision(solution, which="full")
        solution.views.simple.derived.dnn_with_nn_collision = prep_ops.project_full_to_simple(
            solution, solution.views.glob.derived.dnn_with_nn_collision
        )
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
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
    if which == "simple":
        prep_ops.ensure_simple_physical(solution)
        calculate_mfp(solution, which="full")
        solution.views.simple.derived.mfp = prep_ops.project_full_to_simple(solution, solution.views.glob.derived.mfp)
    elif which == "full":
        prep_ops.ensure_simple_physical(solution)
        prep_ops.ensure_full_solution(solution)
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
