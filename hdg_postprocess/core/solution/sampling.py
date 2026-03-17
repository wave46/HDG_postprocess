import numpy as np

from hdg_postprocess.routines.interpolators import SoledgeHDG2DInterpolator
from hdg_postprocess.core.solution import preparation as prep_ops


def calculate_variables_along_line(solution, r_line, z_line, variable_list):
    variable_getters = _line_variable_getters(solution)
    defined_variables = list(variable_getters)
    for variable in variable_list:
        if variable not in defined_variables:
            raise KeyError(f"{variable} is not in the list of posible variables: {defined_variables}")
    result = {}
    for variable in variable_list:
        temp = np.zeros_like(z_line)
        getter = variable_getters[variable]
        for i, (r, z) in enumerate(zip(r_line, z_line)):
            temp[i] = getter(r, z)
        result[variable] = temp
    return result


def save_summary_line(solution, save_folder, r_line, z_line, variable_list):
    defined_variables = list(_line_variable_getters(solution))
    for variable in variable_list:
        if variable not in defined_variables:
            raise KeyError(f"{variable} is not in the list of posible variables: {defined_variables}")
    vertices = np.stack([r_line, z_line]).T
    np.save(f"{save_folder}vertices.npy", vertices)
    values_on_line = calculate_variables_along_line(solution, r_line, z_line, variable_list)
    for variable, values in values_on_line.items():
        np.save(f"{save_folder}{variable}.npy", values)
    return values_on_line


def _line_variable_getters(solution):
    return {
        "n": solution.pointwise.plasma.n,
        "nn": solution.pointwise.plasma.nn,
        "ti": solution.pointwise.plasma.ti,
        "te": solution.pointwise.plasma.te,
        "M": solution.pointwise.plasma.mach,
        "dnn": solution.pointwise.plasma.dnn,
        "mfp": solution.pointwise.plasma.mfp_nn,
        "p_dyn": solution.pointwise.plasma.dynamic_pressure,
        "pi": solution.pointwise.plasma.ion_pressure,
        "dpi_dx": lambda r, z: solution.pointwise.gradients.pi(r, z, "x"),
        "dpi_dy": lambda r, z: solution.pointwise.gradients.pi(r, z, "y"),
        "q_i_par": solution.pointwise.fluxes.ion_heat_parallel,
        "q_i_par_conv": solution.pointwise.fluxes.ion_heat_parallel_convective,
        "q_i_par_cond": solution.pointwise.fluxes.ion_heat_parallel_conductive,
        "q_e_par": solution.pointwise.fluxes.electron_heat_parallel,
        "q_e_par_conv": solution.pointwise.fluxes.electron_heat_parallel_convective,
        "q_e_par_cond": solution.pointwise.fluxes.electron_heat_parallel_conductive,
        "gamma": solution.pointwise.fluxes.particle_parallel,
        "u": solution.pointwise.plasma.u,
        "cs": solution.pointwise.plasma.cs,
        "dk": solution.pointwise.plasma.dk,
        "cx_rate": solution.pointwise.sources.cx_rate,
        "iz_rate": solution.pointwise.sources.ionization_rate,
        "btor": lambda r, z: solution.pointwise.fields.magnetic_field(r, z, "theta"),
        "dbtor_dx": lambda r, z: solution.pointwise.fields.grad_magnetic_field(r, z, "theta", "x"),
        "dbtor_dy": lambda r, z: solution.pointwise.fields.grad_magnetic_field(r, z, "theta", "y"),
        "k": solution.pointwise.plasma.k,
        "psi": solution.pointwise.fields.psi,
        "Q_e_loss_iz": solution.pointwise.sources.Q_e_loss_iz,
        "Q_e_loss_rec": solution.pointwise.sources.Q_e_loss_rec,
        "Q_e_gain_rec": solution.pointwise.sources.Q_e_gain_rec,
        "Q_i_gain_iz": solution.pointwise.sources.Q_i_gain_iz,
        "Q_i_loss_rec": solution.pointwise.sources.Q_i_loss_rec,
        "Q_i_loss_cx": solution.pointwise.sources.Q_i_loss_cx,
        "Q_e_loss_tot": solution.pointwise.sources.Q_e_loss_total,
        "Q_i_loss_tot": solution.pointwise.sources.Q_i_loss_total,
        "Q_loss_tot": solution.pointwise.sources.Q_loss_total,
        "Siz": solution.pointwise.sources.ionization_source,
    }


def define_interpolators(solution):
    prep_ops.ensure_simple_solution(solution)
    prep_ops.ensure_connectivity_big(solution)
    if solution.mesh.metadata.reference_element is None:
        raise ValueError("Please, provide reference element")
    if not solution.mesh.metadata.flags.element_locator_initialized:
        print("Defining an element number mask")
        solution.mesh.geometry.element_locator
    glob_view = solution.views.glob
    if glob_view.equilibrium.qcyl is None:
        solution.equilibrium.define_qcyl(view="glob")
    interpolators = solution.interpolators
    if interpolators.sample is None:
        if solution.mesh.mesh_parameters["element_type"] == "triangle":
            interpolators.sample = SoledgeHDG2DInterpolator(
                solution.mesh.global_state.vertices, np.ones_like(glob_view.solution.conservative[:, :, 0]), solution.mesh.global_state.connectivity,
                solution.mesh.derived_geometry.element_locator, solution.mesh.metadata.reference_element["NodesCoord"],
                solution.mesh.mesh_parameters["element_type"], solution.mesh.metadata.p_order, limit=False,
            )
        elif solution.mesh.mesh_parameters["element_type"] == "quadrilateral":
            interpolators.sample = SoledgeHDG2DInterpolator(
                solution.mesh.global_state.vertices, np.ones_like(glob_view.solution.conservative[:, :, 0]), solution.mesh.global_state.connectivity,
                solution.mesh.derived_geometry.element_locator, solution.mesh.metadata.reference_element["NodesCoord1d"],
                solution.mesh.mesh_parameters["element_type"], solution.mesh.metadata.p_order, limit=False,
            )

    interpolators.solution = []
    interpolators.gradient = []
    for i in range(solution.neq):
        interpolators.solution.append(SoledgeHDG2DInterpolator.instance(interpolators.sample, glob_view.solution.conservative[:, :, i]))
        grad = [
            SoledgeHDG2DInterpolator.instance(interpolators.sample, glob_view.gradient.conservative[:, :, i, 0]),
            SoledgeHDG2DInterpolator.instance(interpolators.sample, glob_view.gradient.conservative[:, :, i, 1]),
        ]
        interpolators.gradient.append(grad)

    interpolators.field = []
    for i in range(3):
        interpolators.field.append(SoledgeHDG2DInterpolator.instance(interpolators.sample, glob_view.equilibrium.magnetic_field[:, :, i]))
    interpolators.qcyl = SoledgeHDG2DInterpolator.instance(interpolators.sample, glob_view.equilibrium.qcyl)
    solution._psi_interpolator = SoledgeHDG2DInterpolator.instance(interpolators.sample, glob_view.equilibrium.poloidal_flux)
