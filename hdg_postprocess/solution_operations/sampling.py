import numpy as np

from hdg_postprocess.routines.interpolators import SoledgeHDG2DInterpolator


def calculate_variables_along_line(solution, r_line, z_line, variable_list):
    defined_variables = [
        "n", "nn", "te", "ti", "M", "dnn", "mfp", "cx_rate", "iz_rate", "u", "cs",
        "p_dyn", "pi", "dpi_dx", "dpi_dy", "q_i_par", "q_e_par", "gamma",
        "q_i_par_conv", "q_i_par_cond", "q_e_par_conv", "q_e_par_cond", "dk",
        "btor", "dbtor_dx", "dbtor_dy", "k", "psi", "Q_e_loss_iz", "Q_e_loss_rec",
        "Q_e_gain_rec", "Q_i_gain_iz", "Q_i_loss_rec", "Q_i_loss_cx", "Q_e_loss_tot",
        "Q_i_loss_tot", "Q_loss_tot", "Siz",
    ]
    for variable in variable_list:
        if variable not in defined_variables:
            raise KeyError(f"{variable} is not in the list of posible variables: {defined_variables}")
    result = {}
    for variable in variable_list:
        temp = np.zeros_like(z_line)
        for i, (r, z) in enumerate(zip(r_line, z_line)):
            if variable == "n":
                temp[i] = solution.pointwise.plasma.n(r, z)
            elif variable == "nn":
                temp[i] = solution.pointwise.plasma.nn(r, z)
            elif variable == "ti":
                temp[i] = solution.pointwise.plasma.ti(r, z)
            elif variable == "te":
                temp[i] = solution.pointwise.plasma.te(r, z)
            elif variable == "M":
                temp[i] = solution.pointwise.plasma.mach(r, z)
            elif variable == "dnn":
                temp[i] = solution.pointwise.plasma.dnn(r, z)
            elif variable == "mfp":
                temp[i] = solution.pointwise.plasma.mfp_nn(r, z)
            elif variable == "p_dyn":
                temp[i] = solution.pointwise.plasma.dynamic_pressure(r, z)
            elif variable == "pi":
                temp[i] = solution.pointwise.plasma.ion_pressure(r, z)
            elif variable == "dpi_dx":
                temp[i] = solution.pointwise.gradients.pi(r, z, "x")
            elif variable == "dpi_dy":
                temp[i] = solution.pointwise.gradients.pi(r, z, "y")
            elif variable == "q_i_par":
                temp[i] = solution.pointwise.fluxes.ion_heat_parallel(r, z)
            elif variable == "q_i_par_conv":
                temp[i] = solution.pointwise.fluxes.ion_heat_parallel_convective(r, z)
            elif variable == "q_i_par_cond":
                temp[i] = solution.pointwise.fluxes.ion_heat_parallel_conductive(r, z)
            elif variable == "q_e_par":
                temp[i] = solution.pointwise.fluxes.electron_heat_parallel(r, z)
            elif variable == "q_e_par_conv":
                temp[i] = solution.pointwise.fluxes.electron_heat_parallel_convective(r, z)
            elif variable == "q_e_par_cond":
                temp[i] = solution.pointwise.fluxes.electron_heat_parallel_conductive(r, z)
            elif variable == "gamma":
                temp[i] = solution.pointwise.fluxes.particle_parallel(r, z)
            elif variable == "u":
                temp[i] = solution.pointwise.plasma.u(r, z)
            elif variable == "cs":
                temp[i] = solution.pointwise.plasma.cs(r, z)
            elif variable == "dk":
                temp[i] = solution.pointwise.plasma.dk(r, z)
            elif variable == "cx_rate":
                temp[i] = solution.pointwise.sources.cx_rate(r, z)
            elif variable == "iz_rate":
                temp[i] = solution.pointwise.sources.ionization_rate(r, z)
            elif variable == "btor":
                temp[i] = solution.pointwise.fields.magnetic_field(r, z, "theta")
            elif variable == "dbtor_dx":
                temp[i] = solution.pointwise.fields.grad_magnetic_field(r, z, "theta", "x")
            elif variable == "dbtor_dy":
                temp[i] = solution.pointwise.fields.grad_magnetic_field(r, z, "theta", "y")
            elif variable == "k":
                temp[i] = solution.pointwise.plasma.k(r, z)
            elif variable == "psi":
                temp[i] = solution.pointwise.fields.psi(r, z)
            elif variable == "Q_e_loss_iz":
                temp[i] = solution.pointwise.sources.Q_e_loss_iz(r, z)
            elif variable == "Q_e_loss_rec":
                temp[i] = solution.pointwise.sources.Q_e_loss_rec(r, z)
            elif variable == "Q_e_gain_rec":
                temp[i] = solution.pointwise.sources.Q_e_gain_rec(r, z)
            elif variable == "Q_i_gain_iz":
                temp[i] = solution.pointwise.sources.Q_i_gain_iz(r, z)
            elif variable == "Q_i_loss_rec":
                temp[i] = solution.pointwise.sources.Q_i_loss_rec(r, z)
            elif variable == "Q_i_loss_cx":
                temp[i] = solution.pointwise.sources.Q_i_loss_cx(r, z)
            elif variable == "Q_e_loss_tot":
                temp[i] = solution.pointwise.sources.Q_e_loss_total(r, z)
            elif variable == "Q_i_loss_tot":
                temp[i] = solution.pointwise.sources.Q_i_loss_total(r, z)
            elif variable == "Q_loss_tot":
                temp[i] = solution.pointwise.sources.Q_loss_total(r, z)
            elif variable == "Siz":
                temp[i] = solution.pointwise.sources.ionization_source(r, z)
            else:
                raise KeyError(f"{variable} is not in the list of posible variables:  {defined_variables}")
        result[variable] = temp
    return result


def save_summary_line(solution, save_folder, r_line, z_line, variable_list):
    defined_variables = [
        "n", "nn", "te", "ti", "M", "dnn", "mfp", "cx_rate", "iz_rate", "u", "cs",
        "p_dyn", "pi", "dpi_dx", "dpi_dy", "q_i_par", "q_e_par", "gamma",
        "q_i_par_conv", "q_i_par_cond", "q_e_par_conv", "q_e_par_cond", "btor",
        "dbtor_dx", "dbtor_dy", "k", "dk", "Q_e_loss_iz", "Q_e_loss_rec",
        "Q_e_gain_rec", "Q_i_gain_iz", "Q_i_loss_rec", "Q_i_loss_cx", "Q_e_loss_tot",
        "Q_i_loss_tot", "Q_loss_tot", "Siz",
    ]
    for variable in variable_list:
        if variable not in defined_variables:
            raise KeyError(f"{variable} is not in the list of posible variables: {defined_variables}")
    vertices = np.stack([r_line, z_line]).T
    np.save(f"{save_folder}vertices.npy", vertices)
    values_on_line = calculate_variables_along_line(solution, r_line, z_line, variable_list)
    for variable, values in values_on_line.items():
        np.save(f"{save_folder}{variable}.npy", values)
    return values_on_line


def define_interpolators(solution):
    if not solution.metadata.flags.combined_simple_solution:
        print("Comibining first simple solution full")
        solution.assembly.simple()
    if solution.mesh.connectivity_big is None:
        print("Comibining first big connectivity")
        solution.mesh.geometry.connectivity_big()
    if solution.mesh.reference_element is None:
        raise ValueError("Please, provide reference element")
    if solution.mesh.element_number is None:
        print("Defining an element number mask")
        solution.mesh.geometry.element_locator()
    glob_view = solution.views.glob
    if glob_view.equilibrium.qcyl is None:
        solution.equilibrium.define_qcyl(view="glob")
    interpolators = solution.interpolators
    if interpolators.sample is None:
        if solution.mesh.mesh_parameters["element_type"] == "triangle":
            interpolators.sample = SoledgeHDG2DInterpolator(
                solution.mesh.vertices_glob, np.ones_like(glob_view.solution.conservative[:, :, 0]), solution.mesh.connectivity_glob,
                solution.mesh.element_number, solution.mesh.reference_element["NodesCoord"],
                solution.mesh.mesh_parameters["element_type"], solution.mesh.p_order, limit=False,
            )
        elif solution.mesh.mesh_parameters["element_type"] == "quadrilateral":
            interpolators.sample = SoledgeHDG2DInterpolator(
                solution.mesh.vertices_glob, np.ones_like(glob_view.solution.conservative[:, :, 0]), solution.mesh.connectivity_glob,
                solution.mesh.element_number, solution.mesh.reference_element["NodesCoord1d"],
                solution.mesh.mesh_parameters["element_type"], solution.mesh.p_order, limit=False,
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
