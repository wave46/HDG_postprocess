import numpy as np

from hdg_postprocess.routines.neutrals import *
from hdg_postprocess.routines.plasma import *
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
    return {
        "boundaries": requested_boundaries,
        "solution": boundary_gauss.solution.conservative,
        "solution_skeleton": boundary_gauss.solution_skeleton.conservative,
        "gradient": boundary_gauss.gradient.conservative,
        "equilibrium": boundary_equilibrium,
        "normal_vector": normal_vector,
        "b_n": np.sum(magnetic_field_unit * normal_vector, axis=-1),
    }


def _evaluate_boundary_summary(solution, boundary_context, variables):
    boundary_solution = boundary_context["solution"]
    boundary_solution_skeleton = boundary_context["solution_skeleton"]
    boundary_gradient = boundary_context["gradient"]
    normal_vector = boundary_context["normal_vector"]
    b_n = boundary_context["b_n"]

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
        "neutral_flux": lambda result: _calculate_neutral_flux(solution, boundary_solution),
        "neutral_flux_skeleton": lambda result: _calculate_neutral_flux(solution, boundary_solution_skeleton),
    }

    result = {}
    for variable in variables:
        if variable in _AUTO_ATTACHED_BOUNDARY_VARIABLES:
            continue
        _evaluate_boundary_variable(variable, result, evaluators)
    return result


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
