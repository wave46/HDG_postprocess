import numpy as np

from hdg_postprocess.routines.atomic import *  # noqa: F403
from hdg_postprocess.routines.neutrals import *  # noqa: F403
from hdg_postprocess.routines.plasma import *  # noqa: F403


def _ensure_interpolators(solution):
    if solution._solution_interpolators is None:
        print("Definition of interpolators will take some time for the initialization")
        solution.define_interpolators()


def _sample_state(solution, r, z):
    _ensure_interpolators(solution)
    state = np.zeros([1, solution.neq])
    for i in range(solution.neq):
        state[0, i] = solution._solution_interpolators[i](r, z)
    return state


def _sample_state_and_gradient(solution, r, z):
    _ensure_interpolators(solution)
    state = np.zeros([1, solution.neq])
    gradient = np.zeros([1, solution.neq, 2])
    for i in range(solution.neq):
        state[0, i] = solution._solution_interpolators[i](r, z)
        for k in range(2):
            gradient[0, i, k] = solution._gradient_interpolators[i][k](r, z)
    return state, gradient


def n(solution, r, z):
    if b"rho" not in solution.parameters["physics"]["physical_variable_names"]:
        raise KeyError("density is not in the models")

    _ensure_interpolators(solution)
    state = np.zeros([1, solution.neq])
    state[:, solution._cons_idx[b"rho"]] = solution._solution_interpolators[solution._cons_idx[b"rho"]](r, z)
    return calculate_n_cons(state, solution.parameters["adimensionalization"]["density_scale"], solution._cons_idx)


def ti(solution, r, z):
    if b"Ti" not in solution.parameters["physics"]["physical_variable_names"]:
        raise KeyError("ion temperature is not in the models")

    _ensure_interpolators(solution)
    state = np.zeros([1, solution.neq])
    state[:, solution._cons_idx[b"rho"]] = solution._solution_interpolators[solution._cons_idx[b"rho"]](r, z)
    if state[:, solution._cons_idx[b"rho"]] == 0:
        return 0
    state[:, solution._cons_idx[b"Gamma"]] = solution._solution_interpolators[solution._cons_idx[b"Gamma"]](r, z)
    state[:, solution._cons_idx[b"nEi"]] = solution._solution_interpolators[solution._cons_idx[b"nEi"]](r, z)
    return calculate_Ti_cons(
        state,
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution._cons_idx,
    )


def te(solution, r, z):
    if b"Te" not in solution.parameters["physics"]["physical_variable_names"]:
        raise KeyError("electron temperature is not in the models")

    _ensure_interpolators(solution)
    state = np.zeros([1, solution.neq])
    state[:, solution._cons_idx[b"rho"]] = solution._solution_interpolators[solution._cons_idx[b"rho"]](r, z)
    if state[:, solution._cons_idx[b"rho"]] == 0:
        return 0
    state[:, solution._cons_idx[b"nEe"]] = solution._solution_interpolators[solution._cons_idx[b"nEe"]](r, z)
    return calculate_Te_cons(
        state,
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution._cons_idx,
    )


def u(solution, r, z):
    if b"u" not in solution.parameters["physics"]["physical_variable_names"]:
        raise KeyError("Mach number is not in the models")

    _ensure_interpolators(solution)
    state = np.zeros([1, solution.neq])
    state[:, solution._cons_idx[b"rho"]] = solution._solution_interpolators[solution._cons_idx[b"rho"]](r, z)
    if state[:, solution._cons_idx[b"rho"]] == 0:
        return 0
    state[:, solution._cons_idx[b"Gamma"]] = solution._solution_interpolators[solution._cons_idx[b"Gamma"]](r, z)
    return calculate_u_cons(state, solution.parameters["adimensionalization"]["speed_scale"], solution._cons_idx)


def cs(solution, r, z):
    if b"Csi" not in solution.parameters["physics"]["physical_variable_names"]:
        raise KeyError("Mach number is not in the models")

    _ensure_interpolators(solution)
    state = np.zeros([1, solution.neq])
    state[:, solution._cons_idx[b"rho"]] = solution._solution_interpolators[solution._cons_idx[b"rho"]](r, z)
    if state[:, solution._cons_idx[b"rho"]] == 0:
        return 0
    state[:, solution._cons_idx[b"Gamma"]] = solution._solution_interpolators[solution._cons_idx[b"Gamma"]](r, z)
    state[:, solution._cons_idx[b"nEi"]] = solution._solution_interpolators[solution._cons_idx[b"nEi"]](r, z)
    state[:, solution._cons_idx[b"nEe"]] = solution._solution_interpolators[solution._cons_idx[b"nEe"]](r, z)
    return calculate_cs_cons(state, solution.parameters["adimensionalization"]["speed_scale"], solution._cons_idx)


def M(solution, r, z):
    if b"M" not in solution.parameters["physics"]["physical_variable_names"]:
        raise KeyError("Mach number is not in the models")

    _ensure_interpolators(solution)
    state = np.zeros([1, solution.neq])
    state[:, solution._cons_idx[b"rho"]] = solution._solution_interpolators[solution._cons_idx[b"rho"]](r, z)
    if state[:, solution._cons_idx[b"rho"]] == 0:
        return 0
    state[:, solution._cons_idx[b"Gamma"]] = solution._solution_interpolators[solution._cons_idx[b"Gamma"]](r, z)
    state[:, solution._cons_idx[b"nEi"]] = solution._solution_interpolators[solution._cons_idx[b"nEi"]](r, z)
    state[:, solution._cons_idx[b"nEe"]] = solution._solution_interpolators[solution._cons_idx[b"nEe"]](r, z)
    return calculate_M_cons(state, solution._cons_idx)


def nn(solution, r, z):
    if b"rhon" not in solution.parameters["physics"]["physical_variable_names"]:
        raise KeyError("neutral density number is not in the models")

    _ensure_interpolators(solution)
    state = np.zeros([1, solution.neq])
    state[:, solution._cons_idx[b"rhon"]] = solution._solution_interpolators[solution._cons_idx[b"rhon"]](r, z)
    return calculate_nn_cons(state, solution.parameters["adimensionalization"]["density_scale"], solution._cons_idx)


def ionization_source_interp(solution, r, z):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if "iz" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide ionization atomic settings for the simulation")
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_iz_source_cons(
        state,
        solution.atomic_parameters["iz"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["physics"]["Mref"],
        solution._cons_idx,
    )


def iz_rate(solution, r, z):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if "iz" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide ionization atomic settings for the simulation")
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_iz_rate_cons(
        state,
        solution.atomic_parameters["iz"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["physics"]["Mref"],
    )


def cx_rate(solution, r, z):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if "cx" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide charge exchange atomic settings for the simulation")
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_cx_rate_cons(
        state,
        solution.atomic_parameters["cx"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
    )


def dnn(solution, r, z):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if "cx" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide charge exchange atomic settings for the simulation")
    if "iz" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide ionization atomic settings for the simulation")
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_dnn_cons(
        state,
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


def k(solution, r, z):
    if b"rho" not in solution.parameters["physics"]["physical_variable_names"]:
        raise KeyError("density is not in the models")

    _ensure_interpolators(solution)
    state = np.zeros([1, solution.neq])
    state[:, solution._cons_idx[b"k"]] = solution._solution_interpolators[solution._cons_idx[b"k"]](r, z)
    return calculate_k_cons(state, solution.parameters["adimensionalization"]["k_scale"], solution._cons_idx)


def dk(solution, r, z):
    if solution.dk_parameters is None:
        raise ValueError("Please, provide turbulent diffusion settings for the simulation")
    _ensure_interpolators(solution)
    if solution.r_axis is None:
        solution.define_minor_radii()

    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    a = np.sqrt((r - solution.r_axis) ** 2 + (z - solution.z_axis) ** 2)
    br = solution._field_interpolators[0](r, z)
    bz = solution._field_interpolators[1](r, z)
    bt = solution._field_interpolators[2](r, z)
    q_cyl = calculate_q_cyl(r, br, bz, bt, a)
    return calculate_dk_cons(
        state,
        solution.dk_parameters,
        q_cyl,
        r / solution.parameters["adimensionalization"]["length_scale"],
        solution.parameters["adimensionalization"]["length_scale"] ** 2
        / solution.parameters["adimensionalization"]["time_scale"],
        solution.cons_idx,
    )


def mfp_nn(solution, r, z):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if "cx" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide charge exchange atomic settings for the simulation")
    if "iz" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide ionization atomic settings for the simulation")
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_mfp_cons(
        state,
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


def p_dyn(solution, r, z):
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_pdyn_cons(
        state,
        (2 / 3 / solution.parameters["physics"]["Mref"])
        * solution.parameters["adimensionalization"]["density_scale"]
        * solution.parameters["adimensionalization"]["temperature_scale"]
        * solution.parameters["adimensionalization"]["charge_scale"],
        solution.parameters["adimensionalization"]["speed_scale"] ** 2
        * solution.parameters["adimensionalization"]["mass_scale"]
        * solution.parameters["adimensionalization"]["density_scale"],
        solution.cons_idx,
    )


def pi(solution, r, z):
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    p0 = (
        (2 / 3 / solution.parameters["physics"]["Mref"])
        * solution.parameters["adimensionalization"]["density_scale"]
        * solution.parameters["adimensionalization"]["temperature_scale"]
        * solution.parameters["adimensionalization"]["charge_scale"]
    )
    return calculate_pi_cons(state, p0, solution.cons_idx)


def grad_ti(solution, r, z, coordinate):
    if coordinate == "x":
        idx = 0
    elif coordinate == "y":
        idx = 1
    else:
        raise ValueError(f"{coordinate} is not a coordinate of the problem")
    state, gradient = _sample_state_and_gradient(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_grad_Ti_cons(
        state,
        gradient,
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["length_scale"],
        solution._cons_idx,
    )[0][idx]


def grad_pi(solution, r, z, coordinate):
    if coordinate == "x":
        idx = 0
    elif coordinate == "y":
        idx = 1
    else:
        raise ValueError(f"{coordinate} is not a coordinate of the problem")
    state, gradient = _sample_state_and_gradient(solution, r, z)
    if state[0, 0] == 0:
        return 0
    p0 = (
        (2 / 3 / solution.parameters["physics"]["Mref"])
        * solution.parameters["adimensionalization"]["density_scale"]
        * solution.parameters["adimensionalization"]["temperature_scale"]
        * solution.parameters["adimensionalization"]["charge_scale"]
    )
    return calculate_grad_pi_cons(
        state,
        gradient,
        p0,
        solution.parameters["adimensionalization"]["length_scale"],
        solution._cons_idx,
    )[0][idx]


def grad_ti_par(solution, r, z):
    state, gradient = _sample_state_and_gradient(solution, r, z)
    if state[0, 0] == 0:
        return 0
    br = solution.field_interpolators[0](r, z)
    bz = solution.field_interpolators[1](r, z)
    bt = solution.field_interpolators[2](r, z)
    return calculate_grad_Ti_par_cons(
        state,
        gradient,
        br,
        bz,
        bt,
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["length_scale"],
        solution._cons_idx,
    )


def grad_te(solution, r, z, coordinate):
    if coordinate == "x":
        idx = 0
    elif coordinate == "y":
        idx = 1
    else:
        raise ValueError(f"{coordinate} is not a coordinate of the problem")
    state, gradient = _sample_state_and_gradient(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_grad_Te_cons(
        state,
        gradient,
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["length_scale"],
        solution._cons_idx,
    )[0][idx]


def grad_te_par(solution, r, z):
    state, gradient = _sample_state_and_gradient(solution, r, z)
    if state[0, 0] == 0:
        return 0
    br = solution._field_interpolators[0](r, z)
    bz = solution._field_interpolators[1](r, z)
    bt = solution._field_interpolators[2](r, z)
    return calculate_grad_Te_par_cons(
        state,
        gradient,
        br,
        bz,
        bt,
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["length_scale"],
        solution._cons_idx,
    )


def particle_flux_par(solution, r, z):
    _ensure_interpolators(solution)
    state = np.zeros([1, solution.neq])
    state[:, solution._cons_idx[b"Gamma"]] = solution._solution_interpolators[solution._cons_idx[b"Gamma"]](r, z)
    return calculate_parallel_flux_cons(
        state,
        solution.parameters["adimensionalization"]["density_scale"]
        * solution.parameters["adimensionalization"]["speed_scale"],
        solution._cons_idx,
    )


def ion_heat_flux_par_conv(solution, r, z):
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_parallel_ion_heat_flux_par_conv_cons(
        state,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution.parameters["adimensionalization"]["mass_scale"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution._cons_idx,
    )


def ion_heat_flux_par_cond(solution, r, z):
    state, gradient = _sample_state_and_gradient(solution, r, z)
    if state[0, 0] == 0:
        return 0
    br = solution._field_interpolators[0](r, z)
    bz = solution._field_interpolators[1](r, z)
    bt = solution._field_interpolators[2](r, z)
    return calculate_parallel_ion_heat_flux_par_cond_cons(
        state,
        gradient,
        br,
        bz,
        bt,
        solution.parameters["physics"]["diff_pari"]
        / (
            solution.parameters["adimensionalization"]["time_scale"] ** 3
            * solution.parameters["adimensionalization"]["temperature_scale"] ** (7 / 2)
            / (
                solution.parameters["adimensionalization"]["density_scale"]
                * solution.parameters["adimensionalization"]["length_scale"] ** 4
            )
            / solution.parameters["adimensionalization"]["mass_scale"]
        ),
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["length_scale"],
        50,
        solution._cons_idx,
    )


def ion_heat_flux_par(solution, r, z):
    state, gradient = _sample_state_and_gradient(solution, r, z)
    if state[0, 0] == 0:
        return 0
    br = solution._field_interpolators[0](r, z)
    bz = solution._field_interpolators[1](r, z)
    bt = solution._field_interpolators[2](r, z)
    return calculate_parallel_ion_heat_flux_par_cons(
        state,
        gradient,
        br,
        bz,
        bt,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["physics"]["diff_pari"]
        / (
            solution.parameters["adimensionalization"]["time_scale"] ** 3
            * solution.parameters["adimensionalization"]["temperature_scale"] ** (7 / 2)
            / (
                solution.parameters["adimensionalization"]["density_scale"]
                * solution.parameters["adimensionalization"]["length_scale"] ** 4
            )
            / solution.parameters["adimensionalization"]["mass_scale"]
        ),
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution.parameters["adimensionalization"]["mass_scale"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution.parameters["adimensionalization"]["length_scale"],
        50,
        solution._cons_idx,
    )


def electron_heat_flux_par_conv(solution, r, z):
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_parallel_electron_heat_flux_par_conv_cons(
        state,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution._cons_idx,
    )


def electron_heat_flux_par_cond(solution, r, z):
    state, gradient = _sample_state_and_gradient(solution, r, z)
    if state[0, 0] == 0:
        return 0
    br = solution._field_interpolators[0](r, z)
    bz = solution._field_interpolators[1](r, z)
    bt = solution._field_interpolators[2](r, z)
    return calculate_parallel_electron_heat_flux_par_cond_cons(
        state,
        gradient,
        br,
        bz,
        bt,
        solution.parameters["physics"]["diff_pare"]
        / (
            solution.parameters["adimensionalization"]["time_scale"] ** 3
            * solution.parameters["adimensionalization"]["temperature_scale"] ** (7 / 2)
            / (
                solution.parameters["adimensionalization"]["density_scale"]
                * solution.parameters["adimensionalization"]["length_scale"] ** 4
            )
            / solution.parameters["adimensionalization"]["mass_scale"]
        ),
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["length_scale"],
        50,
        solution._cons_idx,
    )


def electron_heat_flux_par(solution, r, z):
    state, gradient = _sample_state_and_gradient(solution, r, z)
    if state[0, 0] == 0:
        return 0
    br = solution._field_interpolators[0](r, z)
    bz = solution._field_interpolators[1](r, z)
    bt = solution._field_interpolators[2](r, z)
    return calculate_parallel_electron_heat_flux_par_cons(
        state,
        gradient,
        br,
        bz,
        bt,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["physics"]["diff_pare"]
        / (
            solution.parameters["adimensionalization"]["time_scale"] ** 3
            * solution.parameters["adimensionalization"]["temperature_scale"] ** (7 / 2)
            / (
                solution.parameters["adimensionalization"]["density_scale"]
                * solution.parameters["adimensionalization"]["length_scale"] ** 4
            )
            / solution.parameters["adimensionalization"]["mass_scale"]
        ),
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution.parameters["adimensionalization"]["length_scale"],
        50,
        solution._cons_idx,
    )


def psi(solution, r, z):
    _ensure_interpolators(solution)
    return solution._psi_interpolator(r, z)


def B(solution, r, z, component):
    if component == "R":
        idx = 0
    elif component == "Z":
        idx = 1
    elif component == "theta":
        idx = 2
    else:
        raise ValueError(f"{component} is not a component of the problem")
    _ensure_interpolators(solution)
    return solution._field_interpolators[idx](r, z)


def grad_B(solution, r, z, component, coordinate):
    if component == "R":
        idx = 0
    elif component == "Z":
        idx = 1
    elif component == "theta":
        idx = 2
    else:
        raise ValueError(f"{component} is not a component of the problem")

    if coordinate == "x":
        idx_grad = 0
    elif coordinate == "y":
        idx_grad = 1
    else:
        raise ValueError(f"{coordinate} is not a coordinate of the problem")
    _ensure_interpolators(solution)
    return solution._field_interpolators[idx].gradient(r, z)[idx_grad]


def Q_e_loss_iz(solution, r, z):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if "Eiz" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide Eiz atomic settings for the simulation")
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_electron_sink_due_to_iz_cons(
        state,
        solution.atomic_parameters["Eiz"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution._cons_idx,
    )


def Q_e_loss_rec(solution, r, z):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if "Erec" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide Erec atomic settings for the simulation")
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_electron_sink_due_to_rec_cons(
        state,
        solution.atomic_parameters["Erec"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution._cons_idx,
    )


def Q_e_gain_rec(solution, r, z):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if "rec" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide recombination atomic settings for the simulation")
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_electron_gain_due_to_rec_cons(
        state,
        solution.atomic_parameters["rec"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution._cons_idx,
    )


def Q_e_loss_tot(solution, r, z):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if "Eiz" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide Eiz atomic settings for the simulation")
    if "Erec" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide Erec atomic settings for the simulation")
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_electron_total_loss_cons(
        state,
        solution.atomic_parameters["Eiz"],
        solution.atomic_parameters["Erec"],
        solution.atomic_parameters["rec"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution._cons_idx,
    )


def Q_i_gain_iz(solution, r, z):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if "iz" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide ionization atomic settings for the simulation")
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_ion_gain_due_to_iz_cons(
        state,
        solution.atomic_parameters["iz"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["physics"]["R_E"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution._cons_idx,
    )


def Q_i_loss_rec(solution, r, z):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if "rec" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide recombination atomic settings for the simulation")
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_ion_sink_due_to_rec_cons(
        state,
        solution.atomic_parameters["rec"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["speed_scale"] ** 2
        * solution.parameters["adimensionalization"]["mass_scale"],
    )


def Q_i_loss_cx(solution, r, z):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if "cx" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide charge exchange atomic settings for the simulation")
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_ion_sink_due_to_cx_cons(
        state,
        solution.atomic_parameters["cx"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution.parameters["adimensionalization"]["mass_scale"],
        solution._cons_idx,
    )


def Q_i_loss_tot(solution, r, z):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if "iz" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide ionization atomic settings for the simulation")
    if "rec" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide recombination atomic settings for the simulation")
    if "cx" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide charge exchange atomic settings for the simulation")
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_ion_total_loss_cons(
        state,
        solution.atomic_parameters["iz"],
        solution.atomic_parameters["rec"],
        solution.atomic_parameters["cx"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["physics"]["R_E"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution.parameters["adimensionalization"]["mass_scale"],
        solution.parameters["adimensionalization"]["speed_scale"] ** 2
        * solution.parameters["adimensionalization"]["mass_scale"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution._cons_idx,
    )


def Q_loss_tot(solution, r, z):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if "iz" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide ionization atomic settings for the simulation")
    if "rec" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide recombination atomic settings for the simulation")
    if "cx" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide charge exchange atomic settings for the simulation")
    if "Eiz" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide Eiz atomic settings for the simulation")
    if "Erec" not in solution.atomic_parameters.keys():
        raise ValueError("Please, provide Erec atomic settings for the simulation")
    state = _sample_state(solution, r, z)
    if state[0, 0] == 0:
        return 0
    return calculate_total_loss_cons(
        state,
        solution.atomic_parameters["iz"],
        solution.atomic_parameters["rec"],
        solution.atomic_parameters["cx"],
        solution.atomic_parameters["Eiz"],
        solution.atomic_parameters["Erec"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["physics"]["R_E"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution.parameters["adimensionalization"]["mass_scale"],
        solution.parameters["adimensionalization"]["speed_scale"] ** 2
        * solution.parameters["adimensionalization"]["mass_scale"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution._cons_idx,
    )
