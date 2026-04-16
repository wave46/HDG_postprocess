import numpy as np

from hdg_postprocess.routines.plasma import (
    calculate_Ee_cons,
    calculate_Ei_cons,
    calculate_M_cons,
    calculate_Te_cons,
    calculate_Ti_cons,
    calculate_cs_cons,
    calculate_grad_Ee_cons,
    calculate_grad_Ei_cons,
    calculate_grad_M_cons,
    calculate_grad_Te_cons,
    calculate_grad_Ti_cons,
    calculate_grad_cs_cons,
    calculate_grad_k_cons,
    calculate_grad_n_cons,
    calculate_grad_nn_cons,
    calculate_grad_pe_cons,
    calculate_grad_pi_cons,
    calculate_grad_u_cons,
    calculate_k_cons,
    calculate_n_cons,
    calculate_nn_cons,
    calculate_pe_cons,
    calculate_pi_cons,
    calculate_u_cons,
    calculate_un_cons
)
from hdg_postprocess.core.solution import preparation as prep_ops


def init_phys_variables(solution, which="both"):
    flags = solution.metadata.flags
    if which == "simple":
        prep_ops.ensure_simple_solution(solution)
        simple_view = solution.views.simple
        cons2phys(solution, simple_view.solution.conservative)
        cons2phys(solution, simple_view.gradient.conservative)
        flags.simple_phys_initialized = True
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        glob_view = solution.views.glob
        cons2phys(solution, glob_view.solution.conservative)
        cons2phys(solution, glob_view.gradient.conservative)
        flags.full_phys_initialized = True
    elif which == "gauss":
        if not flags.full_phys_initialized:
            print("Initializing full physical solution first")
            init_phys_variables(solution, which="full")
        prep_ops.ensure_gauss_solution(solution)
        gauss_view = solution.views.gauss
        cons2phys(solution, gauss_view.solution.conservative)
        cons2phys(solution, gauss_view.gradient.conservative)
        flags.gauss_phys_initialized = True
    elif which == "both":
        print("Initializing simple physical solution full")
        init_phys_variables(solution, which="simple")
        print("Initializing full physical solution full")
        init_phys_variables(solution, which="full")


def cons2phys(solution, data):
    if data.shape[-1] == solution.neq:
        if len(data.shape) == 3:
            solution_phys = np.zeros((data.shape[0] * data.shape[1], solution.nphys))
            data_loc = data.reshape((data.shape[0] * data.shape[1], solution.neq))
        elif len(data.shape) == 2:
            solution_phys = np.zeros((data.shape[0], solution.nphys))
            data_loc = data.copy()
        for i in range(solution.nphys):
            phys_variable = solution.parameters["physics"]["physical_variable_names"][i]
            if phys_variable == b"rho":
                solution_phys[:, i] = calculate_n_cons(data_loc, solution.parameters["adimensionalization"]["density_scale"], solution.metadata.indices.conservative)
            elif phys_variable == b"u":
                solution_phys[:, i] = calculate_u_cons(data_loc, solution.parameters["adimensionalization"]["speed_scale"], solution.metadata.indices.conservative)
            elif phys_variable == b"Ei":
                solution_phys[:, i] = calculate_Ei_cons(
                    data_loc,
                    solution.parameters["adimensionalization"]["speed_scale"] ** 2 * solution.parameters["adimensionalization"]["mass_scale"],
                    solution.metadata.indices.conservative,
                )
            elif phys_variable == b"Ee":
                solution_phys[:, i] = calculate_Ee_cons(
                    data_loc,
                    solution.parameters["adimensionalization"]["speed_scale"] ** 2 * solution.parameters["adimensionalization"]["mass_scale"],
                    solution.metadata.indices.conservative,
                )
            elif phys_variable == b"pi":
                solution_phys[:, i] = calculate_pi_cons(
                    data_loc,
                    (2 / 3 / solution.parameters["physics"]["Mref"])
                    * solution.parameters["adimensionalization"]["density_scale"]
                    * solution.parameters["adimensionalization"]["temperature_scale"]
                    * solution.parameters["adimensionalization"]["charge_scale"],
                    solution.metadata.indices.conservative,
                )
            elif phys_variable == b"pe":
                solution_phys[:, i] = calculate_pe_cons(
                    data_loc,
                    (2 / 3 / solution.parameters["physics"]["Mref"])
                    * solution.parameters["adimensionalization"]["density_scale"]
                    * solution.parameters["adimensionalization"]["temperature_scale"]
                    * solution.parameters["adimensionalization"]["charge_scale"],
                    solution.metadata.indices.conservative,
                )
            elif phys_variable == b"Ti":
                solution_phys[:, i] = calculate_Ti_cons(
                    data_loc, solution.parameters["adimensionalization"]["temperature_scale"], solution.parameters["physics"]["Mref"], solution.metadata.indices.conservative
                )
            elif phys_variable == b"Te":
                solution_phys[:, i] = calculate_Te_cons(
                    data_loc, solution.parameters["adimensionalization"]["temperature_scale"], solution.parameters["physics"]["Mref"], solution.metadata.indices.conservative
                )
            elif phys_variable == b"Csi":
                solution_phys[:, i] = calculate_cs_cons(data_loc, solution.parameters["adimensionalization"]["speed_scale"], solution.metadata.indices.conservative)
            elif phys_variable == b"M":
                solution_phys[:, i] = calculate_M_cons(data_loc, solution.metadata.indices.conservative)
            elif phys_variable == b"rhon":
                solution_phys[:, i] = calculate_nn_cons(data_loc, solution.parameters["adimensionalization"]["density_scale"], solution.metadata.indices.conservative)
            elif phys_variable == b"k":
                solution_phys[:, i] = calculate_k_cons(data_loc, solution.parameters["adimensionalization"]["speed_scale"] ** 2, solution.metadata.indices.conservative)
            elif phys_variable == b"un":
                solution_phys[:, i] = calculate_un_cons(data_loc, solution.parameters["adimensionalization"]["density_scale"], solution.metadata.indices.conservative)

            else:
                raise KeyError("Unknown variable, go into the code and add this variable if you are sure")

        _assign_physical_solution(solution, data, solution_phys)

    elif data.shape[-1] == solution.ndim:
        if len(data.shape) == 4:
            grad_phys = np.zeros((data.shape[0] * data.shape[1], solution.nphys, solution.ndim))
            data_loc = data.reshape((data.shape[0] * data.shape[1], solution.neq, solution.ndim))
            if data is solution.views.gauss.gradient.conservative:
                sol_loc = solution.views.gauss.solution.conservative.reshape((data.shape[0] * data.shape[1], solution.neq))
            else:
                sol_loc = solution.views.glob.solution.conservative.reshape((data.shape[0] * data.shape[1], solution.neq))
        elif len(data.shape) == 3:
            grad_phys = np.zeros((data.shape[0], solution.nphys, solution.ndim))
            data_loc = data.copy()
            sol_loc = solution.views.simple.solution.conservative.copy()
        for i in range(solution.nphys):
            phys_variable = solution.parameters["physics"]["physical_variable_names"][i]
            if phys_variable == b"rho":
                grad_phys[:, i, :] = calculate_grad_n_cons(
                    data_loc, solution.parameters["adimensionalization"]["density_scale"], solution.parameters["adimensionalization"]["length_scale"], solution.metadata.indices.conservative
                )
            elif phys_variable == b"u":
                grad_phys[:, i, :] = calculate_grad_u_cons(
                    sol_loc, data_loc, solution.parameters["adimensionalization"]["speed_scale"], solution.parameters["adimensionalization"]["length_scale"], solution.metadata.indices.conservative
                )
            elif phys_variable == b"Ei":
                grad_phys[:, i, :] = calculate_grad_Ei_cons(
                    sol_loc,
                    data_loc,
                    solution.parameters["adimensionalization"]["speed_scale"] ** 2 * solution.parameters["adimensionalization"]["mass_scale"],
                    solution.parameters["adimensionalization"]["length_scale"],
                    solution.metadata.indices.conservative,
                )
            elif phys_variable == b"Ee":
                grad_phys[:, i, :] = calculate_grad_Ee_cons(
                    sol_loc,
                    data_loc,
                    solution.parameters["adimensionalization"]["speed_scale"] ** 2 * solution.parameters["adimensionalization"]["mass_scale"],
                    solution.parameters["adimensionalization"]["length_scale"],
                    solution.metadata.indices.conservative,
                )
            elif phys_variable == b"pi":
                p0 = (2 / 3 / solution.parameters["physics"]["Mref"]) * solution.parameters["adimensionalization"]["density_scale"] * solution.parameters["adimensionalization"]["temperature_scale"] * solution.parameters["adimensionalization"]["charge_scale"]
                grad_phys[:, i, :] = calculate_grad_pi_cons(sol_loc, data_loc, p0, solution.parameters["adimensionalization"]["length_scale"], solution.metadata.indices.conservative)
            elif phys_variable == b"pe":
                grad_phys[:, i, :] = calculate_grad_pe_cons(
                    data_loc,
                    _pressure_scale(solution),
                    solution.parameters["adimensionalization"]["length_scale"],
                    solution.metadata.indices.conservative,
                )
            elif phys_variable == b"Ti":
                grad_phys[:, i, :] = calculate_grad_Ti_cons(
                    sol_loc, data_loc, solution.parameters["adimensionalization"]["temperature_scale"], solution.parameters["physics"]["Mref"], solution.parameters["adimensionalization"]["length_scale"], solution.metadata.indices.conservative
                )
            elif phys_variable == b"Te":
                grad_phys[:, i, :] = calculate_grad_Te_cons(
                    sol_loc, data_loc, solution.parameters["adimensionalization"]["temperature_scale"], solution.parameters["physics"]["Mref"], solution.parameters["adimensionalization"]["length_scale"], solution.metadata.indices.conservative
                )
            elif phys_variable == b"Csi":
                grad_phys[:, i, :] = calculate_grad_cs_cons(
                    sol_loc, data_loc, solution.parameters["adimensionalization"]["speed_scale"], solution.parameters["adimensionalization"]["length_scale"], solution.metadata.indices.conservative
                )
            elif phys_variable == b"M":
                grad_phys[:, i, :] = calculate_grad_M_cons(sol_loc, data_loc, solution.parameters["adimensionalization"]["length_scale"], solution.metadata.indices.conservative)
            elif phys_variable == b"rhon":
                grad_phys[:, i, :] = calculate_grad_nn_cons(
                    data_loc, solution.parameters["adimensionalization"]["density_scale"], solution.parameters["adimensionalization"]["length_scale"], solution.metadata.indices.conservative
                )
            elif phys_variable == b"k":
                grad_phys[:, i, :] = calculate_grad_k_cons(
                    data_loc, solution.parameters["adimensionalization"]["speed_scale"] ** 2, solution.parameters["adimensionalization"]["length_scale"], solution.metadata.indices.conservative
                )

        _assign_physical_gradient(solution, data, grad_phys)


def _assign_physical_solution(solution, data, solution_phys):
    if data is solution.views.glob.solution.conservative or (
        len(data.shape) == 3 and data is not solution.views.gauss.solution.conservative
    ):
        solution.views.glob.solution.physical = solution_phys.reshape((data.shape[0], data.shape[1], solution.nphys))
    elif data is solution.views.gauss.solution.conservative:
        solution.views.gauss.solution.physical = solution_phys.reshape((data.shape[0], data.shape[1], solution.nphys))
    elif len(data.shape) == 2:
        solution.views.simple.solution.physical = solution_phys
    else:
        raise ValueError("Unexpected solution data shape")


def _assign_physical_gradient(solution, data, grad_phys):
    if data is solution.views.glob.gradient.conservative or (
        len(data.shape) == 4 and data is not solution.views.gauss.gradient.conservative
    ):
        solution.views.glob.gradient.physical = grad_phys.reshape((data.shape[0], data.shape[1], solution.nphys, solution.ndim))
    elif data is solution.views.gauss.gradient.conservative:
        solution.views.gauss.gradient.physical = grad_phys.reshape((data.shape[0], data.shape[1], solution.nphys, solution.ndim))
    elif len(data.shape) == 3:
        solution.views.simple.gradient.physical = grad_phys
    else:
        raise ValueError("Unexpected gradient data shape")


def _pressure_scale(solution):
    return (
        (2 / 3 / solution.parameters["physics"]["Mref"])
        * solution.parameters["adimensionalization"]["density_scale"]
        * solution.parameters["adimensionalization"]["temperature_scale"]
        * solution.parameters["adimensionalization"]["charge_scale"]
    )
