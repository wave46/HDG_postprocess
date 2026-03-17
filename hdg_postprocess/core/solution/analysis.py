import numpy as np

from hdg_postprocess.core.solution import preparation as prep_ops
from hdg_postprocess.core.solution import boundary as boundary_ops


def calculate_power_balance(solution):
    """
    Evaluate power balance for the solution.
    """
    power_balance = {}
    print("Calculating volumetric sources for power balance evaluation")
    calculate_volumetric_sources(solution)
    print("Calculating power losses to the wall for power balance evaluation")
    calculate_power_losses_to_wall(solution)

    source_summary = solution.summary.sources
    boundary_summary = solution.summary.boundary
    power_balance["ohmic_heating"] = source_summary.ohmic_source_total
    power_balance["electron_sink_iz"] = source_summary.electron_sink_iz_total
    power_balance["electron_sink_rec"] = source_summary.electron_sink_rec_total
    power_balance["ion_gain_iz"] = source_summary.ion_gain_iz_total
    power_balance["electron_gain_rec"] = source_summary.electron_gain_rec_total
    power_balance["ion_sink_rec"] = source_summary.ion_sink_rec_total
    power_balance["ion_sink_cx"] = source_summary.ion_sink_cx_total
    power_balance["electron_sink_tot"] = (
        source_summary.electron_sink_iz_total
        - source_summary.electron_gain_rec_total
        + source_summary.electron_sink_rec_total
    )
    power_balance["ion_sink_tot"] = (
        source_summary.ion_sink_rec_total
        + source_summary.ion_sink_cx_total
        - source_summary.ion_gain_iz_total
    )
    if "impurity_concentration" in solution.parameters["physics"].keys():
        if solution.parameters["physics"]["impurity_concentration"] > 0:
            power_balance["electron_sink_cooling_factor"] = source_summary.electron_sink_cooling_factor_total
            power_balance["electron_sink_tot"] += source_summary.electron_sink_cooling_factor_total
    power_balance["total_loss"] = power_balance["electron_sink_tot"] + power_balance["ion_sink_tot"]
    if "external_heating" in solution.parameters["physics"].keys():
        power_balance["external_heating"] = source_summary.external_heating_total
        power_balance["total_heating"] = power_balance["ohmic_heating"] + power_balance["external_heating"]
    elif "external_heating_e" in solution.parameters["physics"].keys():
        power_balance["external_heating_e"] = source_summary.external_heating_e_total
        power_balance["external_heating_i"] = source_summary.external_heating_i_total
        power_balance["external_heating"] = (
            power_balance["external_heating_e"] + power_balance["external_heating_i"]
        )
        power_balance["total_heating"] = power_balance["ohmic_heating"] + power_balance["external_heating"]
    else:
        power_balance["total_heating"] = power_balance["ohmic_heating"]

    power_balance["ion_wall_loss"] = boundary_summary.ion_energy_sheath_loss_total
    power_balance["electron_wall_loss"] = boundary_summary.electron_energy_sheath_loss_total
    power_balance["total_wall_loss"] = power_balance["ion_wall_loss"] + power_balance["electron_wall_loss"]
    power_balance["power_balance"] = (
        power_balance["total_heating"] - power_balance["total_loss"] - power_balance["total_wall_loss"]
    )
    power_balance["relative_power_balance"] = power_balance["power_balance"] / power_balance["total_heating"]
    solution._power_balance = power_balance
    return power_balance


def calculate_volumetric_sources(solution):
    prep_ops.ensure_gauss_volumes(solution)
    gauss_sources = solution.views.gauss.sources
    required_sources = [
        ("ohmic_source", "Calculating ohmic source on gauss points first", solution.sources.ohmic),
        ("electron_sink_iz", "Calculating electron ionization sink on gauss points first", solution.sources.electron_sink_iz),
        ("electron_sink_rec", "Calculating electron recombination sink on gauss points first", solution.sources.electron_sink_rec),
        ("ion_gain_iz", "Calculating ionization gain on gauss points first", solution.sources.ion_gain_iz),
        ("electron_gain_rec", "Calculating electron recombination gain on gauss points first", solution.sources.electron_gain_rec),
        ("ion_sink_rec", "Calculating ion recombination sink on gauss points first", solution.sources.ion_sink_rec),
        ("ion_sink_cx", "Calculating ion charge exchange sink on gauss points first", solution.sources.ion_sink_cx),
    ]
    for field_name, message, calculator in required_sources:
        if getattr(gauss_sources, field_name) is None:
            print(message)
            calculator("gauss")

    if "impurity_concentration" in solution.parameters["physics"].keys():
        if solution.parameters["physics"]["impurity_concentration"] > 0:
            if gauss_sources.electron_sink_cooling_factor is None:
                print("Calculating impurity radiation on gauss points first")
                solution.sources.electron_sink_cooling_factor("gauss")

    source_summary = solution.summary.sources
    gauss_volumes = solution.mesh.derived_geometry.gauss_volumes
    source_summary.ohmic_source_total = np.sum(gauss_sources.ohmic_source * gauss_volumes)
    source_summary.electron_sink_iz_total = np.sum(gauss_sources.electron_sink_iz * gauss_volumes)
    source_summary.ion_gain_iz_total = np.sum(gauss_sources.ion_gain_iz * gauss_volumes)
    source_summary.electron_sink_rec_total = np.sum(gauss_sources.electron_sink_rec * gauss_volumes)
    source_summary.electron_gain_rec_total = np.sum(gauss_sources.electron_gain_rec * gauss_volumes)
    source_summary.ion_sink_rec_total = np.sum(gauss_sources.ion_sink_rec * gauss_volumes)
    source_summary.ion_sink_cx_total = np.sum(gauss_sources.ion_sink_cx * gauss_volumes)

    if "external_heating" in solution.parameters["physics"].keys():
        source_summary.external_heating_total = np.sum(gauss_sources.external_heating * gauss_volumes)
    elif "external_heating_e" in solution.parameters["physics"].keys():
        source_summary.external_heating_e_total = np.sum(gauss_sources.external_heating_e * gauss_volumes)
        source_summary.external_heating_i_total = np.sum(gauss_sources.external_heating_i * gauss_volumes)
        source_summary.external_heating_total = (
            source_summary.external_heating_e_total + source_summary.external_heating_i_total
        )
    if "impurity_concentration" in solution.parameters["physics"].keys():
        if solution.parameters["physics"]["impurity_concentration"] > 0:
            source_summary.electron_sink_cooling_factor_total = np.sum(
                gauss_sources.electron_sink_cooling_factor * gauss_volumes
            )


def calculate_power_losses_to_wall(solution):
    boundary_summary = solution.summary.boundary
    if boundary_summary.profile is None:
        print("Calculating boundary summary first")
        boundary_ops.calculate_boundary_summary(solution)

    if boundary_summary.ion_energy_sheath_loss_total is None:
        boundary_summary.ion_energy_sheath_loss_total = (
            boundary_summary.profile["q_i_tot_dep_bc_skeleton"] * boundary_summary.profile["ds"]
        ).sum()
    if boundary_summary.electron_energy_sheath_loss_total is None:
        boundary_summary.electron_energy_sheath_loss_total = (
            boundary_summary.profile["q_e_tot_dep_bc_skeleton"] * boundary_summary.profile["ds"]
        ).sum()
