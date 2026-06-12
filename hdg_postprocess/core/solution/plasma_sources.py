import numpy as np

from hdg_postprocess.routines.atomic import *
from hdg_postprocess.routines.plasma import *
from hdg_postprocess.core.solution.impurity_radiation import get_impurity_radiation_metadata
from hdg_postprocess.core.solution import preparation as prep_ops


def calculate_ohmic_source(solution, which="simple"):
    """
    Calculate the ohmic heating source.
    """
    if "ohmic_coeff" not in solution.parameters["physics"].keys():
        raise KeyError('Please, provide ohmic heating adimensionalized coefficient to self.parameters["physics"]')
    if "Zeff" not in solution.parameters["physics"].keys():
        raise KeyError('Please, effective charge to self.parameters["physics"]')

    if which == "simple":
        calculate_ohmic_source(solution, which="full")
        solution.views.simple.sources.ohmic_source = prep_ops.project_full_to_simple(
            solution, solution.views.glob.sources.ohmic_source
        )
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        glob_view = solution.views.glob
        solution.views.glob.sources.ohmic_source = calculate_ohmic_source_cons(
            glob_view.solution.conservative,
            glob_view.equilibrium.jtor,
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["mass_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["adimensionalization"]["length_scale"],
            solution.parameters["adimensionalization"]["time_scale"],
            solution.parameters["physics"]["ohmic_coeff"],
            solution.parameters["physics"]["Zeff"],
        )
    elif which == "gauss":
        prep_ops.ensure_gauss_solution(solution)
        gauss_view = solution.views.gauss
        gauss_view.sources.ohmic_source = calculate_ohmic_source_cons(
            gauss_view.solution.conservative,
            gauss_view.equilibrium.jtor,
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["mass_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["adimensionalization"]["length_scale"],
            solution.parameters["adimensionalization"]["time_scale"],
            solution.parameters["physics"]["ohmic_coeff"],
            solution.parameters["physics"]["Zeff"],
        )


def calculate_ionization_rate(solution, which="simple"):
    if which == "simple":
        _require_atomic_key(solution, "iz", "Please, provide ionization atomic settings for the simulation")
        prep_ops.ensure_simple_physical(solution)
        calculate_ionization_rate(solution, which="full")
        solution._atomic_rates.ionization_simple = solution._ionization_rate
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        solution._ionization_rate = calculate_iz_rate_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.atomic["iz"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
        )


def calculate_recombination_rate(solution, which="simple"):
    if which == "simple":
        _require_atomic_key(solution, "iz", "Please, provide ionization atomic settings for the simulation")
        prep_ops.ensure_simple_physical(solution)
        calculate_recombination_rate(solution, which="full")
        solution._atomic_rates.recombination_simple = solution._recombination_rate
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        solution._recombination_rate = calculate_rec_rate_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.atomic["rec"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
        )


def calculate_cx_rate(solution, which="simple"):
    if which == "simple":
        _require_atomic_key(solution, "iz", "Please, provide ionization atomic settings for the simulation")
        prep_ops.ensure_simple_physical(solution)
        calculate_cx_rate(solution, which="full")
        solution._atomic_rates.cx_simple = solution._cx_rate
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        solution._cx_rate = calculate_cx_rate_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.atomic["cx"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["physics"]["Mref"],
        )


def calculate_ionization_source(solution, which="simple"):
    _require_atomic_key(solution, "iz", "Please, provide ionization atomic settings for the simulation")
    if which == "simple":
        prep_ops.ensure_simple_physical(solution)
        calculate_ionization_source(solution, which="full")
        _assign_simple_view(solution, "ionization_source", solution.views.glob.sources.ionization_source)
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        solution.views.glob.sources.ionization_source = calculate_iz_source_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.atomic["iz"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution._cons_idx,
        )
    elif which == "gauss":
        prep_ops.ensure_gauss_solution(solution)
        solution.views.gauss.sources.ionization_source = calculate_iz_source_cons(
            solution.views.gauss.solution.conservative,
            solution.additional_parameters.atomic["iz"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution._cons_idx,
        )


def calculate_ion_gain_due_to_iz(solution, which="simple"):
    _require_atomic_key(solution, "iz", "Please, provide ionization atomic settings for the simulation")
    if "R_E" not in solution.parameters["physics"].keys():
        raise ValueError("Please, provide effective energy transfer from neutrals to ions R_E to self.parameters['physics']")
    if which == "simple":
        prep_ops.ensure_simple_physical(solution)
        calculate_ion_gain_due_to_iz(solution, which="full")
        _assign_simple_view(solution, "ion_gain_iz", solution.views.glob.sources.ion_gain_iz)
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        solution.views.glob.sources.ion_gain_iz = calculate_ion_gain_due_to_iz_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.atomic["iz"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["physics"]["R_E"],
            solution.parameters["adimensionalization"]["charge_scale"],
            solution._cons_idx,
        )
    elif which == "gauss":
        prep_ops.ensure_gauss_solution(solution)
        solution.views.gauss.sources.ion_gain_iz = calculate_ion_gain_due_to_iz_cons(
            solution.views.gauss.solution.conservative,
            solution.additional_parameters.atomic["iz"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["physics"]["R_E"],
            solution.parameters["adimensionalization"]["charge_scale"],
            solution._cons_idx,
        )


def calculate_ion_sink_due_to_rec(solution, which="simple"):
    _require_atomic_key(
        solution, "rec", "Please, provide atomic settings for ion losses due to recombination for the simulation"
    )
    if which == "simple":
        prep_ops.ensure_simple_physical(solution)
        calculate_ion_sink_due_to_rec(solution, which="full")
        _assign_simple_view(solution, "ion_sink_rec", solution.views.glob.sources.ion_sink_rec)
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        solution.views.glob.sources.ion_sink_rec = calculate_ion_sink_due_to_rec_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.atomic["rec"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["speed_scale"] ** 2
            * solution.parameters["adimensionalization"]["mass_scale"],
        )
    elif which == "gauss":
        prep_ops.ensure_gauss_solution(solution)
        solution.views.gauss.sources.ion_sink_rec = calculate_ion_sink_due_to_rec_cons(
            solution.views.gauss.solution.conservative,
            solution.additional_parameters.atomic["rec"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["speed_scale"] ** 2
            * solution.parameters["adimensionalization"]["mass_scale"],
        )


def calculate_ion_sink_due_to_cx(solution, which="simple"):
    _require_atomic_key(
        solution, "cx", "Please, provide atomic settings for ion losses due to charge exchange for the simulation"
    )
    if which == "simple":
        prep_ops.ensure_simple_physical(solution)
        calculate_ion_sink_due_to_cx(solution, which="full")
        _assign_simple_view(solution, "ion_sink_cx", solution.views.glob.sources.ion_sink_cx)
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        solution.views.glob.sources.ion_sink_cx = calculate_ion_sink_due_to_cx_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.atomic["cx"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["speed_scale"],
            solution.parameters["adimensionalization"]["mass_scale"],
            solution._cons_idx,
        )
    elif which == "gauss":
        prep_ops.ensure_gauss_solution(solution)
        solution.views.gauss.sources.ion_sink_cx = calculate_ion_sink_due_to_cx_cons(
            solution.views.gauss.solution.conservative,
            solution.additional_parameters.atomic["cx"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["speed_scale"],
            solution.parameters["adimensionalization"]["mass_scale"],
            solution._cons_idx,
        )


def calculate_electron_sink_due_to_iz(solution, which="simple"):
    _require_atomic_key(
        solution, "Eiz", "Please, provide atomic settings for electron losses due to ionization for the simulation"
    )
    if which == "simple":
        prep_ops.ensure_simple_physical(solution)
        calculate_electron_sink_due_to_iz(solution, which="full")
        _assign_simple_view(solution, "electron_sink_iz", solution.views.glob.sources.electron_sink_iz)
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        solution.views.glob.sources.electron_sink_iz = calculate_electron_sink_due_to_iz_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.atomic["Eiz"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["charge_scale"],
            solution._cons_idx,
        )
    elif which == "gauss":
        prep_ops.ensure_gauss_solution(solution)
        solution.views.gauss.sources.electron_sink_iz = calculate_electron_sink_due_to_iz_cons(
            solution.views.gauss.solution.conservative,
            solution.additional_parameters.atomic["Eiz"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["charge_scale"],
            solution._cons_idx,
        )


def calculate_electron_sink_due_to_rec(solution, which="simple"):
    _require_atomic_key(
        solution, "Erec", "Please, provide atomic settings for electron losses due to recombination for the simulation"
    )
    if which == "simple":
        prep_ops.ensure_simple_physical(solution)
        calculate_electron_sink_due_to_rec(solution, which="full")
        _assign_simple_view(solution, "electron_sink_rec", solution.views.glob.sources.electron_sink_rec)
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        solution.views.glob.sources.electron_sink_rec = calculate_electron_sink_due_to_rec_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.atomic["Erec"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["charge_scale"],
            solution._cons_idx,
        )
    elif which == "gauss":
        prep_ops.ensure_gauss_solution(solution)
        solution.views.gauss.sources.electron_sink_rec = calculate_electron_sink_due_to_rec_cons(
            solution.views.gauss.solution.conservative,
            solution.additional_parameters.atomic["Erec"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["charge_scale"],
            solution._cons_idx,
        )


def calculate_electron_gain_due_to_rec(solution, which="simple"):
    _require_atomic_key(solution, "rec", "Please, provide recombination atomic settings for the simulation")
    if which == "simple":
        prep_ops.ensure_simple_physical(solution)
        calculate_electron_gain_due_to_rec(solution, which="full")
        _assign_simple_view(solution, "electron_gain_rec", solution.views.glob.sources.electron_gain_rec)
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        solution.views.glob.sources.electron_gain_rec = calculate_electron_gain_due_to_rec_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.atomic["rec"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["charge_scale"],
            solution._cons_idx,
        )
    elif which == "gauss":
        prep_ops.ensure_gauss_solution(solution)
        solution.views.gauss.sources.electron_gain_rec = calculate_electron_gain_due_to_rec_cons(
            solution.views.gauss.solution.conservative,
            solution.additional_parameters.atomic["rec"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["charge_scale"],
            solution._cons_idx,
        )


def calculate_electron_sink_due_to_cooling_factor(solution, which="simple"):
    if which == "simple":
        prep_ops.ensure_simple_physical(solution)
        calculate_electron_sink_due_to_cooling_factor(solution, which="full")
        _assign_simple_view(solution, "electron_sink_cooling_factor", solution.views.glob.sources.electron_sink_cooling_factor)
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        calculate_impurity_radiation(solution, which="full")
        solution.views.glob.sources.electron_sink_cooling_factor = np.sum(
            solution.views.glob.sources.impurity_radiation,
            axis=-1,
        )
    elif which == "gauss":
        prep_ops.ensure_gauss_solution(solution)
        calculate_impurity_radiation(solution, which="gauss")
        solution.views.gauss.sources.electron_sink_cooling_factor = np.sum(
            solution.views.gauss.sources.impurity_radiation,
            axis=-1,
        )


def calculate_cooling_factor(solution, which="simple"):
    if which == "simple":
        prep_ops.ensure_simple_physical(solution)
        calculate_cooling_factor(solution, which="full")
        _assign_simple_view(solution, "cooling_factor", solution.views.glob.sources.cooling_factor)
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        calculate_impurity_cooling_factors(solution, which="full")
        metadata = get_impurity_radiation_metadata(solution, require_coefficients=True)
        solution.views.glob.sources.cooling_factor = np.sum(
            solution.views.glob.sources.impurity_cooling_factors * metadata.impurity_concentrations,
            axis=-1,
        )
    elif which == "gauss":
        prep_ops.ensure_gauss_solution(solution)
        calculate_impurity_cooling_factors(solution, which="gauss")
        metadata = get_impurity_radiation_metadata(solution, require_coefficients=True)
        solution.views.gauss.sources.cooling_factor = np.sum(
            solution.views.gauss.sources.impurity_cooling_factors * metadata.impurity_concentrations,
            axis=-1,
        )


def calculate_impurity_cooling_factors(solution, which="simple"):
    if which == "simple":
        prep_ops.ensure_simple_physical(solution)
        calculate_impurity_cooling_factors(solution, which="full")
        _assign_simple_view(
            solution,
            "impurity_cooling_factors",
            solution.views.glob.sources.impurity_cooling_factors,
        )
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        solution.views.glob.sources.impurity_cooling_factors = _impurity_cooling_factors_for_solution(
            solution,
            solution.views.glob.solution.conservative,
        )
    elif which == "gauss":
        prep_ops.ensure_gauss_solution(solution)
        solution.views.gauss.sources.impurity_cooling_factors = _impurity_cooling_factors_for_solution(
            solution,
            solution.views.gauss.solution.conservative,
        )


def calculate_impurity_radiation(solution, which="simple"):
    if which == "simple":
        prep_ops.ensure_simple_physical(solution)
        calculate_impurity_radiation(solution, which="full")
        _assign_simple_view(solution, "impurity_radiation", solution.views.glob.sources.impurity_radiation)
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        if solution.views.glob.sources.impurity_cooling_factors is None:
            calculate_impurity_cooling_factors(solution, which="full")
        metadata = get_impurity_radiation_metadata(solution, require_coefficients=True)
        solution.views.glob.sources.impurity_radiation = calculate_impurity_radiation_cons(
            solution.views.glob.solution.conservative,
            solution.views.glob.sources.impurity_cooling_factors,
            metadata.impurity_concentrations,
            _impurity_radiation_source_scale(solution),
        )
    elif which == "gauss":
        prep_ops.ensure_gauss_solution(solution)
        if solution.views.gauss.sources.impurity_cooling_factors is None:
            calculate_impurity_cooling_factors(solution, which="gauss")
        metadata = get_impurity_radiation_metadata(solution, require_coefficients=True)
        solution.views.gauss.sources.impurity_radiation = calculate_impurity_radiation_cons(
            solution.views.gauss.solution.conservative,
            solution.views.gauss.sources.impurity_cooling_factors,
            metadata.impurity_concentrations,
            _impurity_radiation_source_scale(solution),
        )


def calculate_cx_source(solution, which="simple"):
    _require_atomic_key(solution, "iz", "Please, provide ionization atomic settings for the simulation")
    if which == "simple":
        prep_ops.ensure_simple_physical(solution)
        calculate_cx_source(solution, which="full")
        _assign_simple_view(solution, "cx_source", solution.views.glob.sources.cx_source)
    elif which == "full":
        prep_ops.ensure_full_solution(solution)
        solution.views.glob.sources.cx_source = calculate_cx_source_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.atomic["cx"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution._cons_idx,
        )
    elif which == "gauss":
        prep_ops.ensure_gauss_solution(solution)
        solution.views.gauss.sources.cx_source = calculate_cx_source_cons(
            solution.views.gauss.solution.conservative,
            solution.additional_parameters.atomic["cx"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution._cons_idx,
        )


def _require_atomic_key(solution, key, message):
    if solution.additional_parameters.atomic is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if key not in solution.additional_parameters.atomic.keys():
        raise ValueError(message)


def _assign_simple_view(solution, field_name, full_values):
    setattr(solution.views.simple.sources, field_name, prep_ops.project_full_to_simple(solution, full_values))


def _impurity_cooling_factors_for_solution(solution, conservative):
    metadata = get_impurity_radiation_metadata(solution, require_coefficients=True)
    if _uses_saved_impurity_coefficients(solution):
        return calculate_impurity_cooling_factors_cons(
            conservative,
            metadata.impurity_cooling_coefficients_adim,
            solution.parameters["physics"]["Mref"],
        )
    _require_atomic_key(solution, "cooling_factor", "Please, provide atomic settings for the cooling factor for the simulation")
    return calculate_cooling_factor_cons(
        conservative,
        solution.additional_parameters.atomic["cooling_factor"],
        solution.parameters["adimensionalization"]["temperature_scale"],
        solution.parameters["physics"]["Mref"],
        solution.parameters["adimensionalization"]["charge_scale"],
    )[..., None]


def _uses_saved_impurity_coefficients(solution):
    return "impurity_cooling_coefficients_adim" in solution.parameters["physics"]


def _impurity_radiation_source_scale(solution):
    adim = solution.parameters["adimensionalization"]
    if _uses_saved_impurity_coefficients(solution):
        return (
            adim["specific_energy_density_scale"]
            / adim["time_scale"]
            * adim["mass_scale"]
        )
    return adim["charge_scale"] * adim["density_scale"] ** 2
