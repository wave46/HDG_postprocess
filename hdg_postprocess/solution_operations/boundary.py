import numpy as np

from hdg_postprocess.routines.neutrals import *
from hdg_postprocess.routines.plasma import *


def summary_along_the_wall(solution):
    """
    Calculate values in gauss points along the wall.
    """
    if getattr(solution, "_solution_boundary_gauss", None) is None:
        print("Comibining first values on boundary gauss points")
        solution.calculate_in_boundary_gauss_points(np.unique(solution._raw_solution_boundary_infos[0]["boundary_flags"]))

    variables = [
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
    ]

    result = {}
    for variable in variables:
        if variable == "dl":
            res = solution.mesh.segment_length_gauss[:, :, 0]
        elif variable == "ds":
            res = solution.mesh.segment_surface_gauss[:, :, 0]
        elif variable == "normal_vector":
            res = solution.mesh.normals_gauss
        elif variable == "b_n":
            res = np.sum(
                solution.magnetic_field_unit_boundary_gauss[:, :, :2] * solution.mesh.normals_gauss, axis=-1
            )
        elif variable == "solution":
            res = solution.solution_boundary_gauss
        elif variable == "solution_skeleton":
            res = solution.solution_skeleton_boundary_gauss
        elif variable == "gradient":
            res = solution.gradient_boundary_gauss
        elif variable == "n":
            res = calculate_n_cons(
                solution.solution_boundary_gauss,
                solution.parameters["adimensionalization"]["density_scale"],
                solution.cons_idx,
            )
        elif variable == "n_skeleton":
            res = calculate_n_cons(
                solution.solution_skeleton_boundary_gauss,
                solution.parameters["adimensionalization"]["density_scale"],
                solution.cons_idx,
            )
        elif variable == "u":
            res = calculate_u_cons(
                solution.solution_boundary_gauss,
                solution.parameters["adimensionalization"]["speed_scale"],
                solution.cons_idx,
            )
        elif variable == "u_skeleton":
            res = calculate_u_cons(
                solution.solution_skeleton_boundary_gauss,
                solution.parameters["adimensionalization"]["speed_scale"],
                solution.cons_idx,
            )
        elif variable == "te":
            res = calculate_Te_cons(
                solution.solution_boundary_gauss,
                solution.parameters["adimensionalization"]["temperature_scale"],
                solution.parameters["physics"]["Mref"],
                solution._cons_idx,
            )
        elif variable == "te_skeleton":
            res = calculate_Te_cons(
                solution.solution_skeleton_boundary_gauss,
                solution.parameters["adimensionalization"]["temperature_scale"],
                solution.parameters["physics"]["Mref"],
                solution._cons_idx,
            )
        elif variable == "ti":
            res = calculate_Ti_cons(
                solution.solution_boundary_gauss,
                solution.parameters["adimensionalization"]["temperature_scale"],
                solution.parameters["physics"]["Mref"],
                solution._cons_idx,
            )
        elif variable == "ti_skeleton":
            res = calculate_Ti_cons(
                solution.solution_skeleton_boundary_gauss,
                solution.parameters["adimensionalization"]["temperature_scale"],
                solution.parameters["physics"]["Mref"],
                solution._cons_idx,
            )
        elif variable == "M":
            res = calculate_M_cons(solution.solution_boundary_gauss, solution.cons_idx)
        elif variable == "M_skeleton":
            res = calculate_M_cons(solution.solution_skeleton_boundary_gauss, solution.cons_idx)
        elif variable == "p_dyn":
            res = calculate_pdyn_cons(
                solution.solution_boundary_gauss,
                (2 / 3 / solution.parameters["physics"]["Mref"])
                * solution.parameters["adimensionalization"]["density_scale"]
                * solution.parameters["adimensionalization"]["temperature_scale"]
                * solution.parameters["adimensionalization"]["charge_scale"],
                solution.parameters["adimensionalization"]["speed_scale"] ** 2
                * solution.parameters["adimensionalization"]["mass_scale"]
                * solution.parameters["adimensionalization"]["density_scale"],
                solution.cons_idx,
            )
        elif variable == "p_dyn_skeleton":
            res = calculate_pdyn_cons(
                solution.solution_skeleton_boundary_gauss,
                (2 / 3 / solution.parameters["physics"]["Mref"])
                * solution.parameters["adimensionalization"]["density_scale"]
                * solution.parameters["adimensionalization"]["temperature_scale"]
                * solution.parameters["adimensionalization"]["charge_scale"],
                solution.parameters["adimensionalization"]["speed_scale"] ** 2
                * solution.parameters["adimensionalization"]["mass_scale"]
                * solution.parameters["adimensionalization"]["density_scale"],
                solution.cons_idx,
            )
        elif variable == "gamma":
            res = calculate_parallel_flux_cons(
                solution.solution_boundary_gauss,
                solution.parameters["adimensionalization"]["density_scale"]
                * solution.parameters["adimensionalization"]["speed_scale"],
                solution._cons_idx,
            )
        elif variable == "gamma_skeleton":
            res = calculate_parallel_flux_cons(
                solution.solution_skeleton_boundary_gauss,
                solution.parameters["adimensionalization"]["density_scale"]
                * solution.parameters["adimensionalization"]["speed_scale"],
                solution._cons_idx,
            )
        elif variable == "gamma_perp_dep":
            res = _calculate_gamma_perp_dep(solution, solution.solution_boundary_gauss)
        elif variable == "gamma_perp_dep_skeleton":
            res = _calculate_gamma_perp_dep(solution, solution.solution_skeleton_boundary_gauss)
        elif variable == "gamma_tot_dep":
            res = result["gamma"] * result["b_n"] + result["gamma_perp_dep"]
        elif variable == "gamma_tot_dep_skeleton":
            res = result["gamma_skeleton"] * result["b_n"] + result["gamma_perp_dep_skeleton"]
        elif variable == "q_i_par_cond":
            res = _calculate_q_i_par_cond(solution, solution.solution_boundary_gauss)
        elif variable == "q_i_par_cond_skeleton":
            res = _calculate_q_i_par_cond(solution, solution.solution_skeleton_boundary_gauss)
        elif variable == "q_e_par_cond":
            res = _calculate_q_e_par_cond(solution, solution.solution_boundary_gauss)
        elif variable == "q_e_par_cond_skeleton":
            res = _calculate_q_e_par_cond(solution, solution.solution_skeleton_boundary_gauss)
        elif variable == "q_i_par_conv":
            res = _calculate_q_i_par_conv(solution, solution.solution_boundary_gauss)
        elif variable == "q_i_par_conv_skeleton":
            res = _calculate_q_i_par_conv(solution, solution.solution_skeleton_boundary_gauss)
        elif variable == "q_e_par_conv":
            res = _calculate_q_e_par_conv(solution, solution.solution_boundary_gauss)
        elif variable == "q_e_par_conv_skeleton":
            res = _calculate_q_e_par_conv(solution, solution.solution_skeleton_boundary_gauss)
        elif variable == "q_i_par":
            res = _calculate_q_i_par(solution, solution.solution_boundary_gauss)
        elif variable == "q_i_par_skeleton":
            res = _calculate_q_i_par(solution, solution.solution_skeleton_boundary_gauss)
        elif variable == "q_e_par":
            res = _calculate_q_e_par(solution, solution.solution_boundary_gauss)
        elif variable == "q_e_par_skeleton":
            res = _calculate_q_e_par(solution, solution.solution_skeleton_boundary_gauss)
        elif variable == "q_i_perp_dep":
            res = _calculate_q_i_perp_dep(solution, solution.solution_boundary_gauss)
        elif variable == "q_i_perp_dep_skeleton":
            res = _calculate_q_i_perp_dep(solution, solution.solution_skeleton_boundary_gauss)
        elif variable == "q_e_perp_dep":
            res = _calculate_q_e_perp_dep(solution, solution.solution_boundary_gauss)
        elif variable == "q_e_perp_dep_skeleton":
            res = _calculate_q_e_perp_dep(solution, solution.solution_skeleton_boundary_gauss)
        elif variable == "q_i_tot_dep":
            res = result["q_i_par"] * result["b_n"] + result["q_i_perp_dep"]
        elif variable == "q_i_tot_dep_skeleton":
            res = result["q_i_par_skeleton"] * result["b_n"] + result["q_i_perp_dep_skeleton"]
        elif variable == "q_e_tot_dep":
            res = result["q_e_par"] * result["b_n"] + result["q_e_perp_dep"]
        elif variable == "q_e_tot_dep_skeleton":
            res = result["q_e_par_skeleton"] * result["b_n"] + result["q_e_perp_dep_skeleton"]
        elif variable == "q_e_tot_dep_bc":
            res = _calculate_q_e_tot_dep_bc(solution, solution.solution_boundary_gauss)
        elif variable == "q_e_tot_dep_bc_skeleton":
            res = _calculate_q_e_tot_dep_bc(solution, solution.solution_skeleton_boundary_gauss)
        elif variable == "q_i_tot_dep_bc":
            res = _calculate_q_i_tot_dep_bc(solution, solution.solution_boundary_gauss)
        elif variable == "q_i_tot_dep_bc_skeleton":
            res = _calculate_q_i_tot_dep_bc(solution, solution.solution_skeleton_boundary_gauss)
        elif variable == "neutral_flux":
            res = _calculate_neutral_flux(solution, solution.solution_boundary_gauss)
        elif variable == "neutral_flux_skeleton":
            res = _calculate_neutral_flux(solution, solution.solution_skeleton_boundary_gauss)
        result[variable] = res

    for key, item in result.items():
        result[key] = item[:, ::-1]
    result["time"] = solution.parameters["time"]["Current_time"] * solution.parameters["adimensionalization"]["time_scale"]
    result["r"] = solution.mesh.vertices_boundary_gauss[:, ::-1, 0]
    result["z"] = solution.mesh.vertices_boundary_gauss[:, ::-1, 1]
    result["psi"] = solution.poloidal_flux_boundary_gauss[:, ::-1]
    solution._boundary_summary = result
    return result


def calculate_boundary_summary(solution):
    solution.calculate_in_boundary_gauss_points(np.unique(solution._raw_solution_boundary_infos[0]["boundary_flags"]))
    solution._boundary_summary = solution.summary_along_the_wall()
    return solution._boundary_summary


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
    diffusion = _boundary_diffusion(solution, "diff_n") * np.ones_like(boundary_solution[:, :, 0])
    return calculate_particle_perp_flux_wall_cons(
        boundary_solution,
        solution.gradient_boundary_gauss,
        diffusion,
        solution.magnetic_field_boundary_gauss[:, :, 0],
        solution.magnetic_field_boundary_gauss[:, :, 1],
        solution.magnetic_field_boundary_gauss[:, :, 2],
        solution.mesh.normals_gauss,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["length_scale"],
        solution._cons_idx,
    )


def _calculate_q_i_par_cond(solution, boundary_solution):
    return calculate_parallel_ion_heat_flux_par_cond_cons(
        boundary_solution,
        solution.gradient_boundary_gauss,
        solution.magnetic_field_boundary_gauss[:, :, 0],
        solution.magnetic_field_boundary_gauss[:, :, 1],
        solution.magnetic_field_boundary_gauss[:, :, 2],
        _parallel_conductivity(solution, "diff_pari"),
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["length_scale"],
        50,
        solution._cons_idx,
    )


def _calculate_q_e_par_cond(solution, boundary_solution):
    return calculate_parallel_electron_heat_flux_par_cond_cons(
        boundary_solution,
        solution.gradient_boundary_gauss,
        solution.magnetic_field_boundary_gauss[:, :, 0],
        solution.magnetic_field_boundary_gauss[:, :, 1],
        solution.magnetic_field_boundary_gauss[:, :, 2],
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
    return calculate_parallel_ion_heat_flux_par_cons(
        boundary_solution,
        solution.gradient_boundary_gauss,
        solution.magnetic_field_boundary_gauss[:, :, 0],
        solution.magnetic_field_boundary_gauss[:, :, 1],
        solution.magnetic_field_boundary_gauss[:, :, 2],
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
    return calculate_parallel_electron_heat_flux_par_cons(
        boundary_solution,
        solution.gradient_boundary_gauss,
        solution.magnetic_field_boundary_gauss[:, :, 0],
        solution.magnetic_field_boundary_gauss[:, :, 1],
        solution.magnetic_field_boundary_gauss[:, :, 2],
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
        solution.gradient_boundary_gauss,
        diffusion,
        solution.magnetic_field_boundary_gauss[:, :, 0],
        solution.magnetic_field_boundary_gauss[:, :, 1],
        solution.magnetic_field_boundary_gauss[:, :, 2],
        solution.mesh.normals_gauss,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["mass_scale"]
        * solution.parameters["adimensionalization"]["speed_scale"] ** 2,
        solution.parameters["adimensionalization"]["length_scale"],
        solution._cons_idx,
    )


def _calculate_q_e_perp_dep(solution, boundary_solution):
    if solution.parameters["physics"]["diff_n"] != solution.parameters["physics"]["diff_ee"]:
        print("Warning: different perpendicular diffusions and heat conductivities")
        print("Not calculating, providing zeros as perpendicular heat fluxes")
        return np.zeros_like(boundary_solution[:, :, 0])
    diffusion = _boundary_diffusion(solution, "diff_ee") * np.ones_like(boundary_solution[:, :, 0])
    return calculate_perp_electron_heat_wall_cons(
        boundary_solution,
        solution.gradient_boundary_gauss,
        diffusion,
        solution.magnetic_field_boundary_gauss[:, :, 0],
        solution.magnetic_field_boundary_gauss[:, :, 1],
        solution.magnetic_field_boundary_gauss[:, :, 2],
        solution.mesh.normals_gauss,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["mass_scale"]
        * solution.parameters["adimensionalization"]["speed_scale"] ** 2,
        solution.parameters["adimensionalization"]["length_scale"],
        solution._cons_idx,
    )


def _calculate_q_e_tot_dep_bc(solution, boundary_solution):
    return calculate_electron_heat_flux_wall_bc_cons(
        boundary_solution,
        solution.parameters["physics"]["Gmbohme"],
        solution.magnetic_field_boundary_gauss[:, :, 0],
        solution.magnetic_field_boundary_gauss[:, :, 1],
        solution.magnetic_field_boundary_gauss[:, :, 2],
        solution.mesh.normals_gauss,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.e,
        solution._cons_idx,
    )


def _calculate_q_i_tot_dep_bc(solution, boundary_solution):
    return calculate_ion_heat_flux_wall_bc_cons(
        boundary_solution,
        solution.parameters["physics"]["Gmbohm"],
        solution.magnetic_field_boundary_gauss[:, :, 0],
        solution.magnetic_field_boundary_gauss[:, :, 1],
        solution.magnetic_field_boundary_gauss[:, :, 2],
        solution.mesh.normals_gauss,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["speed_scale"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.e,
        solution.parameters["adimensionalization"]["mass_scale"],
        solution._cons_idx,
    )


def _calculate_neutral_flux(solution, boundary_solution):
    return calculate_neutral_perp_flux_wall_cons(
        boundary_solution,
        solution.gradient_boundary_gauss,
        solution.dnn_parameters,
        solution.atomic_parameters,
        solution.magnetic_field_boundary_gauss[:, :, 0],
        solution.magnetic_field_boundary_gauss[:, :, 1],
        solution.magnetic_field_boundary_gauss[:, :, 2],
        solution.mesh.normals_gauss,
        solution.parameters["adimensionalization"]["density_scale"],
        solution.parameters["adimensionalization"]["length_scale"],
        solution.parameters["adimensionalization"]["charge_scale"],
        solution.parameters["adimensionalization"]["mass_scale"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["time_scale"],
        solution._cons_idx,
    )
