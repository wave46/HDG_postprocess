import matplotlib.pyplot as plt
import numpy as np

from hdg_postprocess.routines.neutrals import calculate_dnn_cons
from hdg_postprocess.routines.plasma import (
    calculate_M_cons,
    calculate_Te_cons,
    calculate_Ti_cons,
    calculate_dk_cons,
    calculate_k_cons,
    calculate_n_cons,
    calculate_nn_cons,
)


def _ensure_simple_solution(solution):
    if not solution.combined_simple_solution:
        print("Comibining first simple solution full")
        solution.recombine_simple_full_solution()


def _ensure_simple_physical(solution):
    if not solution.simple_phys_initialized:
        print("Initializing physical solution first")
        solution.init_phys_variables("simple")


def _ensure_connectivity_big(solution):
    if solution.mesh.connectivity_big is None:
        solution.mesh.create_connectivity_big()


def plot_overview(solution, n_levels=100):
    _ensure_simple_solution(solution)
    simple_solution = solution.views.simple.solution.conservative

    solutions_dimensional = simple_solution.copy()
    colorbar_labels = []

    for i in range(solution.neq):
        cons_variable = solution.parameters["physics"]["conservative_variable_names"][i]
        if cons_variable == b"rho":
            solutions_dimensional[:, i] *= solution.parameters["adimensionalization"]["density_scale"]
            colorbar_labels.append(r"n, m$^{-3}$")
            solutions_dimensional[solutions_dimensional[:, i] < 1e8, i] = 1e8
        elif cons_variable == b"Gamma":
            solutions_dimensional[:, i] *= (
                solution.parameters["adimensionalization"]["density_scale"]
                * solution.parameters["adimensionalization"]["speed_scale"]
            )
            colorbar_labels.append(r"$\Gamma$, m$^{-2}$ s$^{-1}$")
        elif cons_variable == b"nEi":
            solutions_dimensional[:, i] *= (
                solution.parameters["adimensionalization"]["density_scale"]
                * solution.parameters["adimensionalization"]["specific_energy_scale"]
            )
            colorbar_labels.append(r"nE$_i$, m$^{-1}$ s$^{-2}$")
        elif cons_variable == b"nEe":
            solutions_dimensional[:, i] *= (
                solution.parameters["adimensionalization"]["density_scale"]
                * solution.parameters["adimensionalization"]["specific_energy_scale"]
            )
            colorbar_labels.append(r"nE$_e$, m$^{-1}$ s$^{-2}$")
        elif cons_variable == b"rhon":
            solutions_dimensional[:, i] *= solution.parameters["adimensionalization"]["density_scale"]
            colorbar_labels.append(r"$n_n$, m$^{-3}$")
            solutions_dimensional[solutions_dimensional[:, i] < 1e8, i] = 1e8
        elif cons_variable == b"k":
            solutions_dimensional[:, i] *= solution.parameters["adimensionalization"]["speed_scale"] ** 2
            colorbar_labels.append(r"$k$, m$^{-2}$/s$^{-2}$")
        else:
            raise NameError("Unknown conservative varibale")

    _ensure_connectivity_big(solution)
    n_lines = int(np.floor(solution.neq / 2 + 0.5))
    fig, axes = plt.subplots(n_lines, 2, figsize=(15, 7.5 * n_lines))

    for i in range(solution.neq):
        cons_variable = solution.parameters["physics"]["conservative_variable_names"][i]
        if (cons_variable != b"Gamma") and (cons_variable != b"k"):
            axes[i // 2, i % 2] = solution.mesh.plot_full_mesh(
                solutions_dimensional[:, i],
                ax=axes[i // 2, i % 2],
                log=True,
                label=colorbar_labels[i],
                connectivity=solution.mesh.connectivity_big,
                n_levels=n_levels,
                cmap="bwr",
            )
        else:
            axes[i // 2, i % 2] = solution.mesh.plot_full_mesh(
                solutions_dimensional[:, i],
                ax=axes[i // 2, i % 2],
                log=False,
                label=colorbar_labels[i],
                connectivity=solution.mesh.connectivity_big,
                n_levels=n_levels,
            )

    return fig, axes, solutions_dimensional


def plot_overview_difference(solution, second_solution, n_levels=100):
    _ensure_simple_solution(solution)
    if not second_solution.combined_simple_solution:
        print("Comibining first simple solution of the second one full")
        second_solution.recombine_simple_full_solution()
    left_simple_solution = solution.views.simple.solution.conservative
    right_simple_solution = second_solution.views.simple.solution.conservative

    difference_dimensional = left_simple_solution.copy() - right_simple_solution.copy()
    colorbar_labels = []
    for i in range(solution.neq):
        cons_variable = solution.parameters["physics"]["conservative_variable_names"][i]
        if cons_variable == b"rho":
            difference_dimensional[:, i] *= solution.parameters["adimensionalization"]["density_scale"]
            colorbar_labels.append(r"n, m$^{-3}$")
        elif cons_variable == b"Gamma":
            difference_dimensional[:, i] *= (
                solution.parameters["adimensionalization"]["density_scale"]
                * solution.parameters["adimensionalization"]["speed_scale"]
            )
            colorbar_labels.append(r"$\Gamma$, m$^{-2}$ s$^{-1}$")
        elif cons_variable == b"nEi":
            difference_dimensional[:, i] *= (
                solution.parameters["adimensionalization"]["density_scale"]
                * solution.parameters["adimensionalization"]["specific_energy_scale"]
            )
            colorbar_labels.append(r"nE$_i$, m$^{-1}$ s$^{-2}$")
        elif cons_variable == b"nEe":
            difference_dimensional[:, i] *= (
                solution.parameters["adimensionalization"]["density_scale"]
                * solution.parameters["adimensionalization"]["specific_energy_scale"]
            )
            colorbar_labels.append(r"nE$_e$, m$^{-1}$ s$^{-2}$")
        elif cons_variable == b"rhon":
            difference_dimensional[:, i] *= solution.parameters["adimensionalization"]["density_scale"]
            colorbar_labels.append(r"$n_n$, m$^{-3}$")
        elif cons_variable == b"k":
            difference_dimensional[:, i] *= solution.parameters["adimensionalization"]["speed_scale"] ** 2
            colorbar_labels.append(r"$k$, m$^{-2}$/s$^{-2}$")
        else:
            raise NameError("Unknown conservative varibale")

    _ensure_connectivity_big(solution)
    n_lines = int(np.floor(solution.neq / 2 + 0.5))
    fig, axes = plt.subplots(n_lines, 2, figsize=(15, 7.5 * n_lines))

    for i in range(solution.neq):
        cons_variable = solution.parameters["physics"]["conservative_variable_names"][i]
        if cons_variable != b"Gamma":
            axes[i // 2, i % 2] = solution.mesh.plot_full_mesh(
                difference_dimensional[:, i],
                ax=axes[i // 2, i % 2],
                log=False,
                label=colorbar_labels[i],
                connectivity=solution.mesh.connectivity_big,
                n_levels=n_levels,
            )
        else:
            axes[i // 2, i % 2] = solution.mesh.plot_full_mesh(
                difference_dimensional[:, i],
                ax=axes[i // 2, i % 2],
                log=False,
                label=colorbar_labels[i],
                connectivity=solution.mesh.connectivity_big,
                n_levels=n_levels,
                cmap="bwr",
            )
    return fig, axes, difference_dimensional


def plot_overview_physical(solution, n_levels=100, limits=None, ticks=None):
    _ensure_simple_physical(solution)

    colorbar_labels = [r"n [m$^{-3}$]", r"$n_n$ [m$^{-3}$]", r"$T_i [eV]$", r"$T_e [eV] $", r"M", r"$k$ [m$^2$/s$^2$]"]
    simple_phys = solution.views.simple.solution.physical
    solutions_plot = np.zeros_like(solution.views.simple.solution.conservative)
    solutions_plot[:, 0] = simple_phys[:, 0]
    solutions_plot[:, 1] = simple_phys[:, -1]
    if solution.neq > 2:
        solutions_plot[:, 2] = simple_phys[:, 6]
        solutions_plot[:, 3] = simple_phys[:, 7]
    if solution.neq > 4:
        solutions_plot[:, 1] = simple_phys[:, 10]
    if solution.neq > 5:
        solutions_plot[:, 5] = simple_phys[:, 11]
    solutions_plot[:, 4] = simple_phys[:, 9]

    _ensure_connectivity_big(solution)
    n_lines = int(np.floor(solution.neq / 2 + 0.5))
    fig, axes = plt.subplots(n_lines, 2, figsize=(15, 7.5 * n_lines))

    for i in range(solution.neq):
        limit = None if limits is None else limits[i]
        tick = None if ticks is None else ticks[i]
        if (i != 4) and (i != 5):
            data = solutions_plot[:, i].copy()
            if (i == 0) or (i == 1):
                data[data < 0] = 1e8
            else:
                data[data < 0] = 1e-3
            axes[i // 2, i % 2] = solution.mesh.plot_full_mesh(
                data,
                ax=axes[i // 2, i % 2],
                log=True,
                label=colorbar_labels[i],
                connectivity=solution.mesh.connectivity_big,
                n_levels=n_levels,
                limits=limit,
                ticks=tick,
            )
        else:
            data = solutions_plot[:, i].copy()
            data[np.where(np.isnan(data))] = 0
            if i == 4:
                axes[i // 2, i % 2] = solution.mesh.plot_full_mesh(
                    data,
                    ax=axes[i // 2, i % 2],
                    log=False,
                    label=colorbar_labels[i],
                    connectivity=solution.mesh.connectivity_big,
                    n_levels=n_levels,
                    limits=limit,
                    cmap="bwr",
                )
            else:
                axes[i // 2, i % 2] = solution.mesh.plot_full_mesh(
                    data,
                    ax=axes[i // 2, i % 2],
                    log=False,
                    label=colorbar_labels[i],
                    connectivity=solution.mesh.connectivity_big,
                    n_levels=n_levels,
                    limits=limit,
                )
    return fig, axes, solutions_plot


def plot_overview_physical_difference(solution, second_solution, n_levels=100):
    _ensure_simple_physical(solution)
    if not second_solution.simple_phys_initialized:
        print("Initializing physical solution first")
        second_solution.init_phys_variables("simple")

    colorbar_labels = [r"n, m$^{-3}$", r"$n_n$, m$^{-3}$", r"$T_i$", r"$T_e$", r"M", r"k"]
    left_simple_phys = solution.views.simple.solution.physical
    right_simple_phys = second_solution.views.simple.solution.physical
    solutions_plot = np.zeros_like(solution.views.simple.solution.conservative)
    solutions_plot[:, 0] = left_simple_phys[:, 0] - right_simple_phys[:, 0]
    solutions_plot[:, 4] = left_simple_phys[:, 9] - right_simple_phys[:, 9]
    if solution.neq > 2:
        solutions_plot[:, 2] = left_simple_phys[:, 6] - right_simple_phys[:, 6]
        solutions_plot[:, 3] = left_simple_phys[:, 7] - right_simple_phys[:, 7]
    if solution.neq > 4:
        solutions_plot[:, 1] = left_simple_phys[:, 10] - right_simple_phys[:, 10]
    if solution.neq > 5:
        solutions_plot[:, 5] = left_simple_phys[:, 11] - right_simple_phys[:, 11]

    _ensure_connectivity_big(solution)
    n_lines = int(np.floor(solution.neq / 2 + 0.5))
    fig, axes = plt.subplots(n_lines, 2, figsize=(15, 7.5 * n_lines))

    for i in range(solution.neq):
        if i == 4:
            axes[i // 2, i % 2] = solution.mesh.plot_full_mesh(
                solutions_plot[:, i],
                ax=axes[i // 2, i % 2],
                log=False,
                label=colorbar_labels[i],
                connectivity=solution.mesh.connectivity_big,
                n_levels=n_levels,
                cmap="bwr",
            )
        else:
            axes[i // 2, i % 2] = solution.mesh.plot_full_mesh(
                solutions_plot[:, i],
                ax=axes[i // 2, i % 2],
                log=False,
                label=colorbar_labels[i],
                connectivity=solution.mesh.connectivity_big,
                n_levels=n_levels,
            )
    return fig, axes, solutions_plot


def plot_variables_overview(solution, variable_list, labels, limits, n_levels, ticks, tick_lables, logs, title=None):
    defined_variables = ["n", "nn", "te", "ti", "M", "dnn", "k", "dk"]
    for variable in variable_list:
        if variable not in defined_variables:
            raise KeyError(f"{variable} is not in the list of posible variables: {defined_variables}")

    _ensure_connectivity_big(solution)
    _ensure_simple_solution(solution)
    simple_solution = solution.views.simple.solution.conservative

    var_to_plot = len(variable_list)
    if var_to_plot == 1:
        fig, axes = plt.subplots(1, 1, figsize=(7.5, 7.5))
    else:
        n_lines = int(np.floor(var_to_plot / 2 + 0.5))
        fig, axes = plt.subplots(n_lines, 2, figsize=(15, 7.5 * n_lines))
    if title is not None:
        fig.suptitle(title)

    res = {}
    for i, (variable, label, limit, tick, tick_label, log) in enumerate(
        zip(variable_list, labels, limits, ticks, tick_lables, logs)
    ):
        if variable == "n":
            data = calculate_n_cons(
                simple_solution, solution.parameters["adimensionalization"]["density_scale"], solution.cons_idx
            )
        elif variable == "nn":
            data = calculate_nn_cons(
                simple_solution, solution.parameters["adimensionalization"]["density_scale"], solution.cons_idx
            )
        elif variable == "te":
            data = calculate_Te_cons(
                simple_solution,
                solution.parameters["adimensionalization"]["temperature_scale"],
                solution.parameters["physics"]["Mref"],
                solution.cons_idx,
            )
        elif variable == "ti":
            data = calculate_Ti_cons(
                simple_solution,
                solution.parameters["adimensionalization"]["temperature_scale"],
                solution.parameters["physics"]["Mref"],
                solution.cons_idx,
            )
        elif variable == "M":
            data = calculate_M_cons(simple_solution, solution.cons_idx)
        elif variable == "dnn":
            data = calculate_dnn_cons(
                simple_solution,
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
        elif variable == "k":
            data = calculate_k_cons(
                simple_solution,
                solution.parameters["adimensionalization"]["speed_scale"] ** 2,
                solution.cons_idx,
            )
        elif variable == "dk":
            if solution.dk_parameters is None:
                raise ValueError("Please, provide turbulent diffusion settings for the simulation")
            if (solution.summary.equilibrium.axis.r is None) or (solution.summary.equilibrium.axis.z is None):
                solution.define_magnetic_axis()
            if solution.views.simple.equilibrium.a is None:
                solution.define_minor_radii(which="simple")
            if solution.views.simple.equilibrium.qcyl is None:
                solution.define_qcyl(which="simple")
            data = calculate_dk_cons(
                simple_solution,
                solution.dk_parameters,
                solution.views.simple.equilibrium.qcyl,
                solution.mesh.vertices_glob[:, 0] / solution.parameters["adimensionalization"]["length_scale"],
                solution.parameters["adimensionalization"]["length_scale"] ** 2
                / solution.parameters["adimensionalization"]["time_scale"],
                solution.cons_idx,
            )
        data[np.isnan(data)] = limit[0]
        if log:
            data[data < 0] = 10.0 ** limit[0]
        res[variable] = data
        cmap = "bwr" if variable == "M" else "jet"
        if var_to_plot > 2:
            axes[i // 2, i % 2] = solution.mesh.plot_full_mesh(
                data,
                ax=axes[i // 2, i % 2],
                log=log,
                label=label,
                connectivity=solution.mesh.connectivity_big,
                n_levels=n_levels,
                ticks=tick,
                tick_labels=tick_label,
                limits=limit,
                cmap=cmap,
            )
        elif var_to_plot == 2:
            axes[i % 2] = solution.mesh.plot_full_mesh(
                data,
                ax=axes[i % 2],
                log=log,
                label=label,
                connectivity=solution.mesh.connectivity_big,
                n_levels=n_levels,
                ticks=tick,
                tick_labels=tick_label,
                limits=limit,
                cmap=cmap,
            )
        else:
            axes = solution.mesh.plot_full_mesh(
                data,
                ax=axes,
                log=log,
                label=label,
                connectivity=solution.mesh.connectivity_big,
                n_levels=n_levels,
                ticks=tick,
                tick_labels=tick_label,
                limits=limit,
                cmap=cmap,
            )
    plt.tight_layout()
    return fig, axes, res
