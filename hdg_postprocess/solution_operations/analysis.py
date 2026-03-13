import numpy as np


def calculate_power_balance(solution):
    """
    Evaluate power balance for the solution.
    """
    power_balance = {}
    print("Calculating volumetric sources for power balance evaluation")
    solution.calculate_volumetric_sources()
    print("Calculating power losses to the wall for power balance evaluation")
    solution.calculate_power_losses_to_wall()

    power_balance["ohmic_heating"] = solution._ohmic_source_total
    power_balance["electron_sink_iz"] = solution._electron_sink_iz_total
    power_balance["electron_sink_rec"] = solution._electron_sink_rec_total
    power_balance["ion_gain_iz"] = solution._ion_gain_iz_total
    power_balance["electron_gain_rec"] = solution._electron_gain_rec_total
    power_balance["ion_sink_rec"] = solution._ion_sink_rec_total
    power_balance["ion_sink_cx"] = solution._ion_sink_cx_total
    power_balance["electron_sink_tot"] = (
        solution._electron_sink_iz_total
        - solution._electron_gain_rec_total
        + solution._electron_sink_rec_total
    )
    power_balance["ion_sink_tot"] = (
        solution._ion_sink_rec_total
        + solution._ion_sink_cx_total
        - solution._ion_gain_iz_total
    )
    if "impurity_concentration" in solution.parameters["physics"].keys():
        if solution.parameters["physics"]["impurity_concentration"] > 0:
            power_balance["electron_sink_cooling_factor"] = solution._electron_sink_cooling_factor_total
            power_balance["electron_sink_tot"] += solution._electron_sink_cooling_factor_total
    power_balance["total_loss"] = power_balance["electron_sink_tot"] + power_balance["ion_sink_tot"]
    if "external_heating" in solution.parameters["physics"].keys():
        power_balance["external_heating"] = solution._external_heating_total
        power_balance["total_heating"] = power_balance["ohmic_heating"] + power_balance["external_heating"]
    elif "external_heating_e" in solution.parameters["physics"].keys():
        power_balance["external_heating_e"] = solution._external_heating_e_total
        power_balance["external_heating_i"] = solution._external_heating_i_total
        power_balance["external_heating"] = (
            power_balance["external_heating_e"] + power_balance["external_heating_i"]
        )
        power_balance["total_heating"] = power_balance["ohmic_heating"] + power_balance["external_heating"]
    else:
        power_balance["total_heating"] = power_balance["ohmic_heating"]

    power_balance["ion_wall_loss"] = solution._ion_energy_sheath_loss_total
    power_balance["electron_wall_loss"] = solution._electron_energy_sheath_loss_total
    power_balance["total_wall_loss"] = power_balance["ion_wall_loss"] + power_balance["electron_wall_loss"]
    power_balance["power_balance"] = (
        power_balance["total_heating"] - power_balance["total_loss"] - power_balance["total_wall_loss"]
    )
    power_balance["relative_power_balance"] = power_balance["power_balance"] / power_balance["total_heating"]
    solution._power_balance = power_balance
    return power_balance


def calculate_volumetric_sources(solution):
    if solution.mesh.volumes_gauss is None:
        solution.mesh.calculate_gauss_volumes()
    if solution._ohmic_source_gauss is None:
        print("Calculating ohmic source on gauss points first")
        solution.calculate_ohmic_source(which="gauss")
    if solution._electron_sink_iz_gauss is None:
        print("Calculating electron ionization sink on gauss points first")
        solution.calculate_electron_sink_due_to_iz(which="gauss")
    if solution._electron_sink_rec_gauss is None:
        print("Calculating electron recombination sink on gauss points first")
        solution.calculate_electron_sink_due_to_rec(which="gauss")
    if solution._ion_gain_iz_gauss is None:
        print("Calculating ionization gain on gauss points first")
        solution.calculate_ion_gain_due_to_iz(which="gauss")
    if solution._electron_gain_rec_gauss is None:
        print("Calculating electron recombination gain on gauss points first")
        solution.calculate_electron_gain_due_to_rec(which="gauss")
    if solution._ion_sink_rec_gauss is None:
        print("Calculating ion recombination sink on gauss points first")
        solution.calculate_ion_sink_due_to_rec(which="gauss")
    if solution._ion_sink_cx_gauss is None:
        print("Calculating ion charge exchange sink on gauss points first")
        solution.calculate_ion_sink_due_to_cx(which="gauss")

    if "impurity_concentration" in solution.parameters["physics"].keys():
        if solution.parameters["physics"]["impurity_concentration"] > 0:
            if solution._electron_sink_cooling_factor_gauss is None:
                print("Calculating impurity radiation on gauss points first")
                solution.calculate_electron_sink_due_to_cooling_factor(which="gauss")

    solution._ohmic_source_total = np.sum(solution._ohmic_source_gauss * solution.mesh.volumes_gauss)
    solution._electron_sink_iz_total = np.sum(solution._electron_sink_iz_gauss * solution.mesh.volumes_gauss)
    solution._ion_gain_iz_total = np.sum(solution._ion_gain_iz_gauss * solution.mesh.volumes_gauss)
    solution._electron_sink_rec_total = np.sum(solution._electron_sink_rec_gauss * solution.mesh.volumes_gauss)
    solution._electron_gain_rec_total = np.sum(solution._electron_gain_rec_gauss * solution.mesh.volumes_gauss)
    solution._ion_sink_rec_total = np.sum(solution._ion_sink_rec_gauss * solution.mesh.volumes_gauss)
    solution._ion_sink_cx_total = np.sum(solution._ion_sink_cx_gauss * solution.mesh.volumes_gauss)

    if "external_heating" in solution.parameters["physics"].keys():
        solution._external_heating_total = np.sum(solution._external_heating_gauss * solution.mesh.volumes_gauss)
    elif "external_heating_e" in solution.parameters["physics"].keys():
        solution._external_heating_e_total = np.sum(
            solution._external_heating_e_gauss * solution.mesh.volumes_gauss
        )
        solution._external_heating_i_total = np.sum(
            solution._external_heating_i_gauss * solution.mesh.volumes_gauss
        )
        solution._external_heating_total = solution._external_heating_e_total + solution._external_heating_i_total
    if "impurity_concentration" in solution.parameters["physics"].keys():
        if solution.parameters["physics"]["impurity_concentration"] > 0:
            solution._electron_sink_cooling_factor_total = np.sum(
                solution._electron_sink_cooling_factor_gauss * solution.mesh.volumes_gauss
            )


def calculate_power_losses_to_wall(solution):
    if solution._boundary_summary is None:
        print("Calculating boundary summary first")
        solution.calculate_boundary_summary()

    if solution._ion_energy_sheath_loss_total is None:
        solution._ion_energy_sheath_loss_total = (
            solution._boundary_summary["q_i_tot_dep_bc_skeleton"] * solution._boundary_summary["ds"]
        ).sum()
    if solution._electron_energy_sheath_loss_total is None:
        solution._electron_energy_sheath_loss_total = (
            solution._boundary_summary["q_e_tot_dep_bc_skeleton"] * solution._boundary_summary["ds"]
        ).sum()
