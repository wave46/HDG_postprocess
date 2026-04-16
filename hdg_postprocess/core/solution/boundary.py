import numpy as np

from hdg_postprocess.routines.neutrals import *
from hdg_postprocess.routines.plasma import *
from hdg_postprocess.routines.tools import double_softplus, softplus
from hdg_postprocess.core.solution import preparation as prep_ops


_GEOMETRY_CORE_VARIABLES = ("r", "z", "psi", "dl", "ds")
_AUTO_ATTACHED_BOUNDARY_VARIABLES = ("r", "z", "psi")
_DEFAULT_BOUNDARY_SUMMARY_VARIABLES = (
    "b_n",
    "normal_vector",
    "dl",
    "ds",
    "solution",
    "solution_skeleton",
    "gradient",
    "n",
    "n_skeleton",
    "u",
    "u_skeleton",
    "te",
    "te_skeleton",
    "ti",
    "ti_skeleton",
    "M",
    "M_skeleton",
    "p_dyn",
    "p_dyn_skeleton",
    "gamma",
    "gamma_skeleton",
    "gamma_perp_dep",
    "gamma_perp_dep_skeleton",
    "gamma_tot_dep",
    "gamma_tot_dep_skeleton",
    "q_i_par_cond",
    "q_i_par_cond_skeleton",
    "q_e_par_cond",
    "q_e_par_cond_skeleton",
    "q_i_par_conv",
    "q_i_par_conv_skeleton",
    "q_e_par_conv",
    "q_e_par_conv_skeleton",
    "q_i_par",
    "q_i_par_skeleton",
    "q_e_par",
    "q_e_par_skeleton",
    "q_i_perp_dep",
    "q_i_perp_dep_skeleton",
    "q_e_perp_dep",
    "q_e_perp_dep_skeleton",
    "q_i_tot_dep",
    "q_i_tot_dep_skeleton",
    "q_e_tot_dep",
    "q_e_tot_dep_skeleton",
    "q_e_tot_dep_bc",
    "q_e_tot_dep_bc_skeleton",
    "q_i_tot_dep_bc",
    "q_i_tot_dep_bc_skeleton",
    "neutral_flux",
    "neutral_flux_skeleton",
)
_BOUNDARY_SUMMARY_DEPENDENCIES = {
    "gamma_tot_dep": ("gamma", "b_n", "gamma_perp_dep"),
    "gamma_tot_dep_skeleton": ("gamma_skeleton", "b_n", "gamma_perp_dep_skeleton"),
    "q_i_tot_dep": ("q_i_par", "b_n", "q_i_perp_dep"),
    "q_i_tot_dep_skeleton": ("q_i_par_skeleton", "b_n", "q_i_perp_dep_skeleton"),
    "q_e_tot_dep": ("q_e_par", "b_n", "q_e_perp_dep"),
    "q_e_tot_dep_skeleton": ("q_e_par_skeleton", "b_n", "q_e_perp_dep_skeleton"),
    "neutral_flux": ("neutral_diff_flux", "neutral_pgrad_flux", "neutral_conv_flux"),
    "neutral_wall_balance": (
        "gamma_parallel_wall",
        "gamma_perp_wall",
        "gamma_pinch_wall",
        "neutral_flux",
        "gamma_puff_wall",
        "gamma_pump_wall",
        "neutral_numerical_flux",
    ),
}

def calculate_boundary_summary(solution, boundaries=None, variables=None):
    """Calculate ordered boundary-Gauss diagnostics for the requested boundaries."""
    boundary_context = _build_boundary_context(solution, boundaries)
    requested_variables = _normalize_requested_boundary_variables(variables)
    result = _evaluate_boundary_summary(solution, boundary_context, requested_variables)
    _attach_boundary_geometry(solution, boundary_context, result)
    solution.summary.boundary.profile = result
    return result


def _build_boundary_context(solution, boundaries=None):
    requested_boundaries = _normalize_requested_boundaries(solution, boundaries)
    if (
        not solution.metadata.flags.combined_boundary_gauss
        or solution.metadata.cache.boundary_gauss_boundaries != requested_boundaries
    ):
        print("Combining boundary gauss values first")
        solution.assembly.boundary_gauss(requested_boundaries)

    boundary_gauss = solution.views.boundary_gauss
    boundary_equilibrium = boundary_gauss.equilibrium
    normal_vector = solution.mesh.boundary_state.normals_gauss
    magnetic_field_unit = boundary_equilibrium.magnetic_field_unit[:, :, :2]
    boundary_flag = _ordered_boundary_flags(solution, requested_boundaries)
    boundary_condition_code = _boundary_condition_codes(solution, boundary_flag)
    n_gauss = boundary_gauss.solution.conservative.shape[1]
    return {
        "boundaries": requested_boundaries,
        "solution": boundary_gauss.solution.conservative,
        "solution_skeleton": boundary_gauss.solution_skeleton.conservative,
        "gradient": boundary_gauss.gradient.conservative,
        "equilibrium": boundary_equilibrium,
        "normal_vector": normal_vector,
        "b_n": np.sum(magnetic_field_unit * normal_vector, axis=-1),
        "boundary_flag": np.repeat(boundary_flag[:, None], n_gauss, axis=1),
        "boundary_condition_code": np.repeat(boundary_condition_code[:, None], n_gauss, axis=1),
    }


def _evaluate_boundary_summary(solution, boundary_context, variables):
    boundary_solution = boundary_context["solution"]
    boundary_solution_skeleton = boundary_context["solution_skeleton"]
    boundary_gradient = boundary_context["gradient"]
    normal_vector = boundary_context["normal_vector"]
    b_n = boundary_context["b_n"]
    boundary_flag = boundary_context["boundary_flag"]
    boundary_condition_code = boundary_context["boundary_condition_code"]

    p_dyn_scale = (
        (2 / 3 / solution.parameters["physics"]["Mref"])
        * solution.parameters["adimensionalization"]["density_scale"]
        * solution.parameters["adimensionalization"]["temperature_scale"]
        * solution.parameters["adimensionalization"]["charge_scale"]
    )
    p_dyn_mass_scale = (
        solution.parameters["adimensionalization"]["speed_scale"] ** 2
        * solution.parameters["adimensionalization"]["mass_scale"]
        * solution.parameters["adimensionalization"]["density_scale"]
    )
    evaluators = {
        "dl": lambda result: solution.mesh.boundary_state.segment_length_gauss[:, :, 0],
        "ds": lambda result: solution.mesh.boundary_state.segment_surface_gauss[:, :, 0],
        "normal_vector": lambda result: normal_vector,
        "b_n": lambda result: b_n,
        "boundary_flag": lambda result: boundary_flag,
        "boundary_condition_code": lambda result: boundary_condition_code,
        "solution": lambda result: boundary_solution,
        "solution_skeleton": lambda result: boundary_solution_skeleton,
        "gradient": lambda result: boundary_gradient,
        "n": lambda result: calculate_n_cons(
            boundary_solution,
            solution.parameters["adimensionalization"]["density_scale"],
            solution.metadata.indices.conservative,
        ),
        "n_skeleton": lambda result: calculate_n_cons(
            boundary_solution_skeleton,
            solution.parameters["adimensionalization"]["density_scale"],
            solution.metadata.indices.conservative,
        ),
        "u": lambda result: calculate_u_cons(
            boundary_solution,
            solution.parameters["adimensionalization"]["speed_scale"],
            solution.metadata.indices.conservative,
        ),
        "u_skeleton": lambda result: calculate_u_cons(
            boundary_solution_skeleton,
            solution.parameters["adimensionalization"]["speed_scale"],
            solution.metadata.indices.conservative,
        ),
        "te": lambda result: calculate_Te_cons(
            boundary_solution,
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["physics"]["Mref"],
            solution._cons_idx,
        ),
        "te_skeleton": lambda result: calculate_Te_cons(
            boundary_solution_skeleton,
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["physics"]["Mref"],
            solution._cons_idx,
        ),
        "ti": lambda result: calculate_Ti_cons(
            boundary_solution,
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["physics"]["Mref"],
            solution._cons_idx,
        ),
        "ti_skeleton": lambda result: calculate_Ti_cons(
            boundary_solution_skeleton,
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["physics"]["Mref"],
            solution._cons_idx,
        ),
        "M": lambda result: calculate_M_cons(boundary_solution, solution.metadata.indices.conservative),
        "M_skeleton": lambda result: calculate_M_cons(
            boundary_solution_skeleton,
            solution.metadata.indices.conservative,
        ),
        "p_dyn": lambda result: calculate_pdyn_cons(
            boundary_solution,
            p_dyn_scale,
            p_dyn_mass_scale,
            solution.metadata.indices.conservative,
        ),
        "p_dyn_skeleton": lambda result: calculate_pdyn_cons(
            boundary_solution_skeleton,
            p_dyn_scale,
            p_dyn_mass_scale,
            solution.metadata.indices.conservative,
        ),
        "gamma": lambda result: calculate_parallel_flux_cons(
            boundary_solution,
            solution.parameters["adimensionalization"]["density_scale"]
            * solution.parameters["adimensionalization"]["speed_scale"],
            solution._cons_idx,
        ),
        "gamma_skeleton": lambda result: calculate_parallel_flux_cons(
            boundary_solution_skeleton,
            solution.parameters["adimensionalization"]["density_scale"]
            * solution.parameters["adimensionalization"]["speed_scale"],
            solution._cons_idx,
        ),
        "gamma_perp_dep": lambda result: _calculate_gamma_perp_dep(solution, boundary_solution),
        "gamma_perp_dep_skeleton": lambda result: _calculate_gamma_perp_dep(solution, boundary_solution_skeleton),
        "gamma_tot_dep": lambda result: result["gamma"] * result["b_n"] + result["gamma_perp_dep"],
        "gamma_tot_dep_skeleton": lambda result: result["gamma_skeleton"] * result["b_n"] + result["gamma_perp_dep_skeleton"],
        "q_i_par_cond": lambda result: _calculate_q_i_par_cond(solution, boundary_solution),
        "q_i_par_cond_skeleton": lambda result: _calculate_q_i_par_cond(solution, boundary_solution_skeleton),
        "q_e_par_cond": lambda result: _calculate_q_e_par_cond(solution, boundary_solution),
        "q_e_par_cond_skeleton": lambda result: _calculate_q_e_par_cond(solution, boundary_solution_skeleton),
        "q_i_par_conv": lambda result: _calculate_q_i_par_conv(solution, boundary_solution),
        "q_i_par_conv_skeleton": lambda result: _calculate_q_i_par_conv(solution, boundary_solution_skeleton),
        "q_e_par_conv": lambda result: _calculate_q_e_par_conv(solution, boundary_solution),
        "q_e_par_conv_skeleton": lambda result: _calculate_q_e_par_conv(solution, boundary_solution_skeleton),
        "q_i_par": lambda result: _calculate_q_i_par(solution, boundary_solution),
        "q_i_par_skeleton": lambda result: _calculate_q_i_par(solution, boundary_solution_skeleton),
        "q_e_par": lambda result: _calculate_q_e_par(solution, boundary_solution),
        "q_e_par_skeleton": lambda result: _calculate_q_e_par(solution, boundary_solution_skeleton),
        "q_i_perp_dep": lambda result: _calculate_q_i_perp_dep(solution, boundary_solution),
        "q_i_perp_dep_skeleton": lambda result: _calculate_q_i_perp_dep(solution, boundary_solution_skeleton),
        "q_e_perp_dep": lambda result: _calculate_q_e_perp_dep(solution, boundary_solution),
        "q_e_perp_dep_skeleton": lambda result: _calculate_q_e_perp_dep(solution, boundary_solution_skeleton),
        "q_i_tot_dep": lambda result: result["q_i_par"] * result["b_n"] + result["q_i_perp_dep"],
        "q_i_tot_dep_skeleton": lambda result: result["q_i_par_skeleton"] * result["b_n"] + result["q_i_perp_dep_skeleton"],
        "q_e_tot_dep": lambda result: result["q_e_par"] * result["b_n"] + result["q_e_perp_dep"],
        "q_e_tot_dep_skeleton": lambda result: result["q_e_par_skeleton"] * result["b_n"] + result["q_e_perp_dep_skeleton"],
        "q_e_tot_dep_bc": lambda result: _calculate_q_e_tot_dep_bc(solution, boundary_solution),
        "q_e_tot_dep_bc_skeleton": lambda result: _calculate_q_e_tot_dep_bc(solution, boundary_solution_skeleton),
        "q_i_tot_dep_bc": lambda result: _calculate_q_i_tot_dep_bc(solution, boundary_solution),
        "q_i_tot_dep_bc_skeleton": lambda result: _calculate_q_i_tot_dep_bc(solution, boundary_solution_skeleton),
        "gamma_parallel_wall": lambda result: _calculate_gamma_parallel_wall(solution, boundary_context),
        "gamma_perp_wall": lambda result: _calculate_gamma_perp_wall(solution, boundary_context),
        "gamma_pinch_wall": lambda result: _calculate_gamma_pinch_wall(solution, boundary_context),
        "gamma_puff_wall": lambda result: _calculate_gamma_puff_wall(solution, boundary_context),
        "gamma_pump_wall": lambda result: _calculate_gamma_pump_wall(solution, boundary_context),
        "neutral_diff_flux": lambda result: _calculate_neutral_diff_flux(solution, boundary_context),
        "neutral_pgrad_flux": lambda result: _calculate_neutral_pgrad_flux(solution, boundary_context),
        "neutral_conv_flux": lambda result: _calculate_neutral_conv_flux(solution, boundary_context),
        "neutral_flux": lambda result: result["neutral_diff_flux"] + result["neutral_pgrad_flux"] + result["neutral_conv_flux"],
        "neutral_numerical_flux": lambda result: _calculate_neutral_numerical_flux(solution, boundary_context),
        "neutral_wall_balance": lambda result: (
            result["gamma_parallel_wall"]
            - result["gamma_perp_wall"]
            - result["gamma_pinch_wall"]
            - result["neutral_flux"]
            + result["gamma_puff_wall"]
            - result["gamma_pump_wall"]
            + result["neutral_numerical_flux"]
        ),
        "neutral_flux_skeleton": lambda result: _calculate_neutral_flux(solution, boundary_solution_skeleton),
    }

    result = {}
    for variable in variables:
        if variable in _AUTO_ATTACHED_BOUNDARY_VARIABLES:
            continue
        _evaluate_boundary_variable(variable, result, evaluators)
    return {key: result[key] for key in variables if key not in _AUTO_ATTACHED_BOUNDARY_VARIABLES}


def _evaluate_boundary_variable(variable, result, evaluators):
    if variable in result:
        return result[variable]
    for dependency in _BOUNDARY_SUMMARY_DEPENDENCIES.get(variable, ()):
        _evaluate_boundary_variable(dependency, result, evaluators)
    result[variable] = evaluators[variable](result)
    return result[variable]


def _attach_boundary_geometry(solution, boundary_context, result):
    for key, item in list(result.items()):
        result[key] = item[:, ::-1]
    result["time"] = solution.parameters["time"]["Current_time"] * solution.parameters["adimensionalization"]["time_scale"]
    result["r"] = solution.mesh.boundary_state.vertices_gauss[:, ::-1, 0]
    result["z"] = solution.mesh.boundary_state.vertices_gauss[:, ::-1, 1]
    result["psi"] = boundary_context["equilibrium"].poloidal_flux[:, ::-1]


def _normalize_requested_boundary_variables(variables):
    if variables is None:
        return _DEFAULT_BOUNDARY_SUMMARY_VARIABLES
    requested = list(dict.fromkeys(_GEOMETRY_CORE_VARIABLES + tuple(variables)))
    return tuple(requested)


def _normalize_requested_boundaries(solution, boundaries):
    if boundaries is None:
        return tuple(np.unique(solution.raw.boundary_infos[0]["boundary_flags"]).tolist())
    return tuple(np.asarray(boundaries, dtype=int).tolist())


def _ensure_default_boundary_gauss(solution):
    default_boundaries = _normalize_requested_boundaries(solution, None)
    if (
        not solution.metadata.flags.combined_boundary_gauss
        or solution.metadata.cache.boundary_gauss_boundaries != default_boundaries
    ):
        print("Combining boundary gauss values first")
        solution.assembly.boundary_gauss(default_boundaries)


def _ordered_boundary_flags(solution, boundaries):
    row_flags = []
    for boundary_index, component_index in solution.metadata.cache.boundary_gauss_ordering:
        boundary_id = boundaries[boundary_index]
        component = solution.mesh.boundary_state.connectivity[boundary_id][component_index]
        row_flags.extend([boundary_id] * component.shape[0])
    return np.asarray(row_flags, dtype=int)


def _boundary_condition_codes(solution, boundary_flags):
    codes = np.zeros_like(boundary_flags, dtype=int)
    boundary_map = np.asarray(solution.parameters["physics"].get("boundary_flags", []), dtype=int).reshape(-1)
    valid = (boundary_flags > 0) & (boundary_flags <= boundary_map.size)
    codes[valid] = boundary_map[boundary_flags[valid] - 1]
    return codes


def _boundary_flux_scale(solution):
    adim = solution.parameters["adimensionalization"]
    return adim["density_scale"] * adim["speed_scale"]


def _parameter_scalar(container, key, default=0.0):
    value = container.get(key, default)
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return float(default)
    return float(array.reshape(-1)[0])


def _boundary_diffusion_adim(solution, key):
    return _boundary_diffusion(solution, key) / (
        solution.parameters["adimensionalization"]["length_scale"] ** 2
        / solution.parameters["adimensionalization"]["time_scale"]
    )


def _boundary_recycling_coeff(solution, boundary_context):
    codes = boundary_context["boundary_condition_code"]
    physics = solution.parameters["physics"]
    recycling = _parameter_scalar(physics, "recycling", 0.0)
    recycling_pump = _parameter_scalar(physics, "recycling_pump", recycling)
    coeff = np.zeros_like(codes, dtype=float)
    coeff[(codes == 50) | (codes == 56)] = recycling
    coeff[codes == 55] = recycling_pump
    return coeff


def _require_atomic_parameters(solution):
    atomic = solution.parameters["physics"].get("atomic")
    if atomic is None:
        atomic = solution.additional_parameters.atomic
    if atomic is None:
        raise ValueError(
            "Atomic parameters are required for Bohm-wall neutral diagnostics. "
            "Configure them with solution.additional_parameters.set_atomic(...) or load them from the solution file."
        )
    return atomic


def _limited_ti_adim(solution, boundary_solution):
    temperature_scale = solution.parameters["adimensionalization"]["temperature_scale"]
    ti = calculate_Ti_cons(
        boundary_solution,
        temperature_scale,
        solution.parameters["physics"]["Mref"],
        solution._cons_idx,
    ) / temperature_scale
    return softplus(ti, 1e-6 / temperature_scale, 0.01, 10)


def _sigmavnn_cons(solution, boundary_solution):
    ti = calculate_Ti_cons(
        boundary_solution,
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution._cons_idx,
    )
    kb = 1.38064852e-23
    s0 = 5.2958e-17
    thermal = np.maximum(ti * solution.metadata.constants.elemental_charge / kb, 0.0)
    return s0 * thermal ** 0.25


def _neutral_dnn_adim(solution, boundary_solution):
    atomic = _require_atomic_parameters(solution)
    adim = solution.parameters["adimensionalization"]
    physics = solution.parameters["physics"]
    inn = solution._cons_idx[b"rhon"]

    ti_limited = _limited_ti_adim(solution, boundary_solution)
    coeff = (
        adim["charge_scale"]
        * adim["temperature_scale"]
        * ti_limited
        / (adim["mass_scale"] * adim["density_scale"])
        * adim["time_scale"]
        / adim["length_scale"] ** 2
    )
    sigmaviz = calculate_iz_rate_cons(
        boundary_solution,
        atomic["iz"],
        adim["temperature_scale"],
        adim["density_scale"],
        physics["Mref"],
    )
    sigmavcx = calculate_cx_rate_cons(
        boundary_solution,
        atomic["cx"],
        adim["temperature_scale"],
        physics["Mref"],
    )
    sigmavnn = _sigmavnn_cons(solution, boundary_solution)
    denom = boundary_solution[:, :, 0] * (sigmaviz + sigmavcx) + boundary_solution[:, :, inn] * sigmavnn
    raw_dnn = np.divide(coeff, denom, out=np.zeros_like(coeff), where=np.abs(denom) > 0)
    diff_n = _boundary_diffusion_adim(solution, "diff_n")
    diff_nn = _parameter_scalar(physics, "diff_nn", 0.0)
    if diff_nn <= 0.0 and solution.additional_parameters.neutral_diffusion is not None:
        diff_nn = float(solution.additional_parameters.neutral_diffusion["dnn_max_adim"])
    if diff_nn <= 0.0:
        diff_nn = 10.0 * diff_n
    return double_softplus(raw_dnn, 10.0 * diff_n, diff_nn, 0.01, 10)


def _neutral_w5p(solution, boundary_solution):
    ti_limited = _limited_ti_adim(solution, boundary_solution)
    ti_supp = 1e-6 / solution.parameters["adimensionalization"]["temperature_scale"]
    ti_factor = 2.0 / (3.0 * solution.parameters["physics"]["Mref"])
    neutralp_lambda = _parameter_scalar(solution.parameters["numerics"], "NeutralP_lambda", 0.0)
    if neutralp_lambda == 0.0:
        return np.zeros_like(boundary_solution, dtype=float)

    inn = solution._cons_idx[b"rhon"]
    alpha = neutralp_lambda * ti_factor * boundary_solution[:, :, inn] * _neutral_dnn_adim(solution, boundary_solution)
    alpha = np.divide(alpha, ti_limited, out=np.zeros_like(alpha), where=np.abs(ti_limited) > 0)
    supp = np.divide(ti_limited, ti_limited + ti_supp, out=np.zeros_like(ti_limited), where=np.abs(ti_limited + ti_supp) > 0)

    w5p = np.zeros_like(boundary_solution, dtype=float)
    rho = boundary_solution[:, :, 0]
    gamma = boundary_solution[:, :, 1]
    n_ei = boundary_solution[:, :, 2]

    rho_sq = rho ** 2
    rho_cu = rho ** 3
    valid_sq = np.abs(rho_sq) > 0
    valid_cu = np.abs(rho_cu) > 0

    w5p[:, :, 0] = np.divide(gamma ** 2, rho_cu, out=np.zeros_like(alpha), where=valid_cu) - np.divide(
        n_ei, rho_sq, out=np.zeros_like(alpha), where=valid_sq
    )
    w5p[:, :, 1] = -np.divide(gamma, rho_sq, out=np.zeros_like(alpha), where=valid_sq)
    w5p[:, :, 2] = np.divide(1.0, rho, out=np.zeros_like(alpha), where=np.abs(rho) > 0)
    return alpha[:, :, None] * supp[:, :, None] * w5p


def _tau_neutral(solution):
    numerics = solution.parameters["numerics"]
    inn = solution._cons_idx[b"rhon"]
    if "tau" in numerics:
        tau = np.asarray(numerics["tau"], dtype=float).reshape(-1)
        if tau.size > inn:
            return float(tau[inn])
    if "Stabilization_parameter" in numerics:
        tau = np.asarray(numerics["Stabilization_parameter"], dtype=float).reshape(-1)
        if tau.size > inn:
            return float(tau[inn])
    return 1.0


def _puff_area(solution, boundary_context):
    mask = boundary_context["boundary_condition_code"] == 56
    if not np.any(mask):
        return 0.0
    return float(np.sum(solution.mesh.boundary_state.segment_surface_gauss[:, :, 0][mask]))


def _pump_area(solution, boundary_context):
    mask = boundary_context["boundary_condition_code"] == 55
    if not np.any(mask):
        return 0.0
    return float(np.sum(solution.mesh.boundary_state.segment_surface_gauss[:, :, 0][mask]))


def _parallel_conductivity(solution, key):
    return solution.parameters["physics"][key] / (
        solution.parameters["adimensionalization"]["time_scale"] ** 3
        * solution.parameters["adimensionalization"]["temperature_scale"] ** (7 / 2)
        / (
            solution.parameters["adimensionalization"]["density_scale"]
            * solution.parameters["adimensionalization"]["length_scale"] ** 4
        )
        / solution.parameters["adimensionalization"]["mass_scale"]
    )


def _boundary_diffusion(solution, key):
    physics = solution.parameters["physics"]
    if f"ME_{key}" in physics:
        key = f"ME_{key}"
    return (
        physics[key]
        * solution.parameters["adimensionalization"]["length_scale"] ** 2
        / solution.parameters["adimensionalization"]["time_scale"]
    )


def _calculate_gamma_perp_dep(solution, boundary_solution):
    boundary_gauss = solution.views.boundary_gauss
    diffusion = _boundary_diffusion(solution, "diff_n") * np.ones_like(boundary_solution[:, :, 0])
    return calculate_particle_perp_flux_wall_cons(
        boundary_solution,
        boundary_gauss.gradient.conservative,
        diffusion,
        boundary_gauss.equilibrium.magnetic_field[:, :, 0],
        boundary_gauss.equilibrium.magnetic_field[:, :, 1],
        boundary_gauss.equilibrium.magnetic_field[:, :, 2],
        solution.mesh.boundary_state.normals_gauss,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["length_scale"],
        solution._cons_idx,
    )


def _calculate_q_i_par_cond(solution, boundary_solution):
    boundary_gauss = solution.views.boundary_gauss
    return calculate_parallel_ion_heat_flux_par_cond_cons(
        boundary_solution,
        boundary_gauss.gradient.conservative,
        boundary_gauss.equilibrium.magnetic_field[:, :, 0],
        boundary_gauss.equilibrium.magnetic_field[:, :, 1],
        boundary_gauss.equilibrium.magnetic_field[:, :, 2],
        _parallel_conductivity(solution, "diff_pari"),
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["length_scale"],
        50,
        solution._cons_idx,
    )


def _calculate_q_e_par_cond(solution, boundary_solution):
    boundary_gauss = solution.views.boundary_gauss
    return calculate_parallel_electron_heat_flux_par_cond_cons(
        boundary_solution,
        boundary_gauss.gradient.conservative,
        boundary_gauss.equilibrium.magnetic_field[:, :, 0],
        boundary_gauss.equilibrium.magnetic_field[:, :, 1],
        boundary_gauss.equilibrium.magnetic_field[:, :, 2],
        _parallel_conductivity(solution, "diff_pare"),
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["length_scale"],
        50,
        solution._cons_idx,
    )


def _calculate_q_i_par_conv(solution, boundary_solution):
    return calculate_parallel_ion_heat_flux_par_conv_cons(
        boundary_solution,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution.parameters["adimensionalization"]["mass_scale"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution._cons_idx,
    )


def _calculate_q_e_par_conv(solution, boundary_solution):
    return calculate_parallel_electron_heat_flux_par_conv_cons(
        boundary_solution,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution._cons_idx,
    )


def _calculate_q_i_par(solution, boundary_solution):
    boundary_gauss = solution.views.boundary_gauss
    return calculate_parallel_ion_heat_flux_par_cons(
        boundary_solution,
        boundary_gauss.gradient.conservative,
        boundary_gauss.equilibrium.magnetic_field[:, :, 0],
        boundary_gauss.equilibrium.magnetic_field[:, :, 1],
        boundary_gauss.equilibrium.magnetic_field[:, :, 2],
        solution.parameters["adimensionalization"]["density_scale"],
        _parallel_conductivity(solution, "diff_pari"),
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution.parameters["adimensionalization"]["mass_scale"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution.parameters["adimensionalization"]["length_scale"],
        50,
        solution._cons_idx,
    )


def _calculate_q_e_par(solution, boundary_solution):
    boundary_gauss = solution.views.boundary_gauss
    return calculate_parallel_electron_heat_flux_par_cons(
        boundary_solution,
        boundary_gauss.gradient.conservative,
        boundary_gauss.equilibrium.magnetic_field[:, :, 0],
        boundary_gauss.equilibrium.magnetic_field[:, :, 1],
        boundary_gauss.equilibrium.magnetic_field[:, :, 2],
        solution.parameters["adimensionalization"]["density_scale"],
        _parallel_conductivity(solution, "diff_pare"),
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution.parameters["adimensionalization"]["length_scale"],
        50,
        solution._cons_idx,
    )


def _calculate_q_i_perp_dep(solution, boundary_solution):
    boundary_gauss = solution.views.boundary_gauss
    if (
        (solution.parameters["physics"]["diff_n"] != solution.parameters["physics"]["diff_e"])
        or (solution.parameters["physics"]["diff_e"] != solution.parameters["physics"]["diff_u"])
    ):
        print("Warning: different perpendicular diffusions and heat conductivities")
        print("Not calculating, providing zeros as perpendicular heat fluxes")
        return np.zeros_like(boundary_solution[:, :, 0])
    diffusion = _boundary_diffusion(solution, "diff_e") * np.ones_like(boundary_solution[:, :, 0])
    return calculate_perp_ion_heat_wall_cons(
        boundary_solution,
        boundary_gauss.gradient.conservative,
        diffusion,
        boundary_gauss.equilibrium.magnetic_field[:, :, 0],
        boundary_gauss.equilibrium.magnetic_field[:, :, 1],
        boundary_gauss.equilibrium.magnetic_field[:, :, 2],
        solution.mesh.boundary_state.normals_gauss,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["mass_scale"]
        * solution.parameters["adimensionalization"]["speed_scale"] ** 2,
        solution.parameters["adimensionalization"]["length_scale"],
        solution._cons_idx,
    )


def _calculate_q_e_perp_dep(solution, boundary_solution):
    boundary_gauss = solution.views.boundary_gauss
    if solution.parameters["physics"]["diff_n"] != solution.parameters["physics"]["diff_ee"]:
        print("Warning: different perpendicular diffusions and heat conductivities")
        print("Not calculating, providing zeros as perpendicular heat fluxes")
        return np.zeros_like(boundary_solution[:, :, 0])
    diffusion = _boundary_diffusion(solution, "diff_ee") * np.ones_like(boundary_solution[:, :, 0])
    return calculate_perp_electron_heat_wall_cons(
        boundary_solution,
        boundary_gauss.gradient.conservative,
        diffusion,
        boundary_gauss.equilibrium.magnetic_field[:, :, 0],
        boundary_gauss.equilibrium.magnetic_field[:, :, 1],
        boundary_gauss.equilibrium.magnetic_field[:, :, 2],
        solution.mesh.boundary_state.normals_gauss,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["mass_scale"]
        * solution.parameters["adimensionalization"]["speed_scale"] ** 2,
        solution.parameters["adimensionalization"]["length_scale"],
        solution._cons_idx,
    )


def _calculate_q_e_tot_dep_bc(solution, boundary_solution):
    boundary_gauss = solution.views.boundary_gauss
    return calculate_electron_heat_flux_wall_bc_cons(
        boundary_solution,
        solution.parameters["physics"]["Gmbohme"],
        boundary_gauss.equilibrium.magnetic_field[:, :, 0],
        boundary_gauss.equilibrium.magnetic_field[:, :, 1],
        boundary_gauss.equilibrium.magnetic_field[:, :, 2],
        solution.mesh.boundary_state.normals_gauss,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.metadata.constants.elemental_charge,
        solution._cons_idx,
    )


def _calculate_q_i_tot_dep_bc(solution, boundary_solution):
    boundary_gauss = solution.views.boundary_gauss
    return calculate_ion_heat_flux_wall_bc_cons(
        boundary_solution,
        solution.parameters["physics"]["Gmbohm"],
        boundary_gauss.equilibrium.magnetic_field[:, :, 0],
        boundary_gauss.equilibrium.magnetic_field[:, :, 1],
        boundary_gauss.equilibrium.magnetic_field[:, :, 2],
        solution.mesh.boundary_state.normals_gauss,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.metadata.constants.elemental_charge,
        solution.parameters["adimensionalization"]["mass_scale"],
        solution._cons_idx,
    )


def _calculate_neutral_flux(solution, boundary_solution):
    boundary_gauss = solution.views.boundary_gauss
    return calculate_neutral_perp_flux_wall_cons(
        boundary_solution,
        boundary_gauss.gradient.conservative,
        solution.additional_parameters.neutral_diffusion,
        solution.additional_parameters.atomic,
        boundary_gauss.equilibrium.magnetic_field[:, :, 0],
        boundary_gauss.equilibrium.magnetic_field[:, :, 1],
        boundary_gauss.equilibrium.magnetic_field[:, :, 2],
        solution.mesh.boundary_state.normals_gauss,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["length_scale"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution.parameters["adimensionalization"]["mass_scale"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["time_scale"],
        solution._cons_idx,
    )


def _calculate_gamma_parallel_wall(solution, boundary_context):
    gamma_skeleton = calculate_parallel_flux_cons(
        boundary_context["solution_skeleton"],
        _boundary_flux_scale(solution),
        solution._cons_idx,
    )
    return _boundary_recycling_coeff(solution, boundary_context) * gamma_skeleton * boundary_context["b_n"]


def _calculate_gamma_perp_wall(solution, boundary_context):
    boundary_equilibrium = boundary_context["equilibrium"]
    diffusion = _boundary_diffusion(solution, "diff_n") * np.ones_like(boundary_context["solution"][:, :, 0])
    gamma_perp_dep = calculate_particle_perp_flux_wall_cons(
        boundary_context["solution"],
        boundary_context["gradient"],
        diffusion,
        boundary_equilibrium.magnetic_field[:, :, 0],
        boundary_equilibrium.magnetic_field[:, :, 1],
        boundary_equilibrium.magnetic_field[:, :, 2],
        boundary_context["normal_vector"],
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["length_scale"],
        solution._cons_idx,
    )
    return -_boundary_recycling_coeff(solution, boundary_context) * gamma_perp_dep


def _calculate_gamma_pinch_wall(solution, boundary_context):
    if not solution.transport_1d.available:
        return np.zeros_like(boundary_context["solution"][:, :, 0])

    profiles = solution.transport_1d.effective_profiles(dimensional=True)
    if "vpinch_fs" not in profiles:
        return np.zeros_like(boundary_context["solution"][:, :, 0])

    rho_grid = np.asarray(profiles["rho_grid"], dtype=float)
    vpinch_fs = np.asarray(profiles["vpinch_fs"], dtype=float)
    rho_local = np.sqrt(np.maximum(boundary_context["equilibrium"].poloidal_flux, 0.0))
    vpinch = np.interp(np.clip(rho_local, rho_grid[0], rho_grid[-1]), rho_grid, vpinch_fs)
    bpol = boundary_context["equilibrium"].magnetic_field[:, :, :2]
    bpol_norm = np.linalg.norm(bpol, axis=-1)
    bpol_unit = np.divide(bpol, bpol_norm[:, :, None], out=np.zeros_like(bpol), where=bpol_norm[:, :, None] > 0)
    apinch_dot_n = vpinch * (
        bpol_unit[:, :, 1] * boundary_context["normal_vector"][:, :, 0]
        - bpol_unit[:, :, 0] * boundary_context["normal_vector"][:, :, 1]
    )
    neutral_density = calculate_n_cons(
        boundary_context["solution_skeleton"],
        solution.parameters["adimensionalization"]["density_scale"],
        solution._cons_idx,
    )
    return -_boundary_recycling_coeff(solution, boundary_context) * neutral_density * apinch_dot_n


def _calculate_gamma_puff_wall(solution, boundary_context):
    puff_rate = _parameter_scalar(solution.parameters["physics"], "puff", 0.0)
    if puff_rate == 0.0:
        return np.zeros_like(boundary_context["solution"][:, :, 0])

    puff_area = _puff_area(solution, boundary_context)
    result = np.zeros_like(boundary_context["solution"][:, :, 0], dtype=float)
    if puff_area > 0.0:
        result[boundary_context["boundary_condition_code"] == 56] = puff_rate / puff_area
    return result


def _calculate_gamma_pump_wall(solution, boundary_context):
    cryopump_power = _parameter_scalar(solution.parameters["physics"], "cryopump_power", 10.0)
    if cryopump_power == 0.0:
        return np.zeros_like(boundary_context["solution"][:, :, 0])

    pump_area = _pump_area(solution, boundary_context)
    if pump_area <= 0.0:
        return np.zeros_like(boundary_context["solution"][:, :, 0])

    neutral_density = calculate_nn_cons(
        boundary_context["solution"],
        solution.parameters["adimensionalization"]["density_scale"],
        solution._cons_idx,
    )
    result = np.zeros_like(neutral_density, dtype=float)
    mask = boundary_context["boundary_condition_code"] == 55
    result[mask] = cryopump_power * neutral_density[mask] / pump_area
    return result


def _calculate_neutral_diff_flux(solution, boundary_context):
    dnn = _neutral_dnn_adim(solution, boundary_context["solution"])
    return _boundary_flux_scale(solution) * dnn * (
        boundary_context["gradient"][:, :, solution._cons_idx[b"rhon"], 0] * boundary_context["normal_vector"][:, :, 0]
        + boundary_context["gradient"][:, :, solution._cons_idx[b"rhon"], 1] * boundary_context["normal_vector"][:, :, 1]
    )


def _calculate_neutral_pgrad_flux(solution, boundary_context):
    qn = np.sum(
        boundary_context["gradient"] * boundary_context["normal_vector"][:, :, None, :],
        axis=-1,
    )
    return _boundary_flux_scale(solution) * np.sum(qn * _neutral_w5p(solution, boundary_context["solution"]), axis=-1)


def _calculate_neutral_conv_flux(solution, boundary_context):
    inn = solution._cons_idx[b"rhon"]
    rho = boundary_context["solution"][:, :, solution._cons_idx[b"rho"]]
    gamma = boundary_context["solution"][:, :, solution._cons_idx[b"Gamma"]]
    rhon = boundary_context["solution"][:, :, inn]

    convection = np.zeros_like(rho, dtype=float)
    

    if b"Gamman" in solution._cons_idx:
        convection = convection - boundary_context["solution"][:, :, solution._cons_idx[b"Gamman"]]
    return _boundary_flux_scale(solution) * convection * boundary_context["b_n"]


def _calculate_neutral_numerical_flux(solution, boundary_context):
    inn = solution._cons_idx[b"rhon"]
    jump = boundary_context["solution_skeleton"][:, :, inn] - boundary_context["solution"][:, :, inn]
    return _tau_neutral(solution) * _boundary_flux_scale(solution) * jump
