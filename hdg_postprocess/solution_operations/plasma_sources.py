import numpy as np

from hdg_postprocess.routines.atomic import *
from hdg_postprocess.routines.plasma import *


def calculate_ohmic_source(solution, which="simple"):
    """
    Calculate the ohmic heating source.
    """
    if "ohmic_coeff" not in solution.parameters["physics"].keys():
        raise KeyError('Please, provide ohmic heating adimensionalized coefficient to self.parameters["physics"]')
    if "Zeff" not in solution.parameters["physics"].keys():
        raise KeyError('Please, effective charge to self.parameters["physics"]')

    if which == "simple":
        solution.calculate_ohmic_source(which="full")
        glob_solution = solution.views.glob.solution.conservative
        solution.views.simple.sources.ohmic_source = np.zeros(solution.mesh.vertices_glob.shape[0])
        solution.views.simple.sources.ohmic_source[
            solution.mesh.connectivity_glob.reshape(-1, 1).ravel()
        ] = solution.views.glob.sources.ohmic_source.reshape(glob_solution.shape[0] * glob_solution.shape[1])
    elif which == "full":
        _ensure_full_solution(solution)
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
        _ensure_full_solution(solution)
        if solution.views.gauss.equilibrium.jtor is None:
            print("Calculating on gauss points first")
            solution.calculate_in_gauss_points()
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
        _ensure_simple_phys(solution)
        solution.calculate_ionization_rate(which="full")
        solution._atomic_rates.ionization_simple = solution._ionization_rate
    elif which == "full":
        _ensure_full_solution(solution)
        solution._ionization_rate = calculate_iz_rate_cons(
            solution.views.glob.solution.conservative,
            solution.atomic_parameters["iz"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
        )


def calculate_recombination_rate(solution, which="simple"):
    if which == "simple":
        _require_atomic_key(solution, "iz", "Please, provide ionization atomic settings for the simulation")
        _ensure_simple_phys(solution)
        solution.calculate_recombination_rate(which="full")
        solution._atomic_rates.recombination_simple = solution._recombination_rate
    elif which == "full":
        _ensure_full_solution(solution)
        solution._recombination_rate = calculate_rec_rate_cons(
            solution.views.glob.solution.conservative,
            solution.atomic_parameters["rec"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
        )


def calculate_cx_rate(solution, which="simple"):
    if which == "simple":
        _require_atomic_key(solution, "iz", "Please, provide ionization atomic settings for the simulation")
        _ensure_simple_phys(solution)
        solution.calculate_cx_rate(which="full")
        solution._atomic_rates.cx_simple = solution._cx_rate
    elif which == "full":
        _ensure_full_solution(solution)
        solution._cx_rate = calculate_cx_rate_cons(
            solution.views.glob.solution.conservative,
            solution.atomic_parameters["cx"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["physics"]["Mref"],
        )


def calculate_ionization_source(solution, which="simple"):
    _require_atomic_key(solution, "iz", "Please, provide ionization atomic settings for the simulation")
    if which == "simple":
        _ensure_simple_phys(solution)
        solution.calculate_ionization_source(which="full")
        _assign_simple_view(solution, "ionization_source", solution.views.glob.sources.ionization_source)
    elif which == "full":
        _ensure_full_solution(solution)
        solution.views.glob.sources.ionization_source = calculate_iz_source_cons(
            solution.views.glob.solution.conservative,
            solution.atomic_parameters["iz"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution._cons_idx,
        )
    elif which == "gauss":
        _ensure_gauss_solution(solution)
        solution.views.gauss.sources.ionization_source = calculate_iz_source_cons(
            solution.views.gauss.solution.conservative,
            solution.atomic_parameters["iz"],
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
        _ensure_simple_phys(solution)
        solution.calculate_ion_gain_due_to_iz(which="full")
        _assign_simple_view(solution, "ion_gain_iz", solution.views.glob.sources.ion_gain_iz)
    elif which == "full":
        _ensure_full_solution(solution)
        solution.views.glob.sources.ion_gain_iz = calculate_ion_gain_due_to_iz_cons(
            solution.views.glob.solution.conservative,
            solution.atomic_parameters["iz"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["physics"]["R_E"],
            solution.parameters["adimensionalization"]["charge_scale"],
            solution._cons_idx,
        )
    elif which == "gauss":
        _ensure_gauss_solution(solution)
        solution.views.gauss.sources.ion_gain_iz = calculate_ion_gain_due_to_iz_cons(
            solution.views.gauss.solution.conservative,
            solution.atomic_parameters["iz"],
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
        _ensure_simple_phys(solution)
        solution.calculate_ion_sink_due_to_rec(which="full")
        _assign_simple_view(solution, "ion_sink_rec", solution.views.glob.sources.ion_sink_rec)
    elif which == "full":
        _ensure_full_solution(solution)
        solution.views.glob.sources.ion_sink_rec = calculate_ion_sink_due_to_rec_cons(
            solution.views.glob.solution.conservative,
            solution.atomic_parameters["rec"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["speed_scale"] ** 2
            * solution.parameters["adimensionalization"]["mass_scale"],
        )
    elif which == "gauss":
        _ensure_gauss_solution(solution)
        solution.views.gauss.sources.ion_sink_rec = calculate_ion_sink_due_to_rec_cons(
            solution.views.gauss.solution.conservative,
            solution.atomic_parameters["rec"],
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
        _ensure_simple_phys(solution)
        solution.calculate_ion_sink_due_to_cx(which="full")
        _assign_simple_view(solution, "ion_sink_cx", solution.views.glob.sources.ion_sink_cx)
    elif which == "full":
        _ensure_full_solution(solution)
        solution.views.glob.sources.ion_sink_cx = calculate_ion_sink_due_to_cx_cons(
            solution.views.glob.solution.conservative,
            solution.atomic_parameters["cx"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["speed_scale"],
            solution.parameters["adimensionalization"]["mass_scale"],
            solution._cons_idx,
        )
    elif which == "gauss":
        _ensure_gauss_solution(solution)
        solution.views.gauss.sources.ion_sink_cx = calculate_ion_sink_due_to_cx_cons(
            solution.views.gauss.solution.conservative,
            solution.atomic_parameters["cx"],
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
        _ensure_simple_phys(solution)
        solution.calculate_electron_sink_due_to_iz(which="full")
        _assign_simple_view(solution, "electron_sink_iz", solution.views.glob.sources.electron_sink_iz)
    elif which == "full":
        _ensure_full_solution(solution)
        solution.views.glob.sources.electron_sink_iz = calculate_electron_sink_due_to_iz_cons(
            solution.views.glob.solution.conservative,
            solution.atomic_parameters["Eiz"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["charge_scale"],
            solution._cons_idx,
        )
    elif which == "gauss":
        _ensure_gauss_solution(solution)
        solution.views.gauss.sources.electron_sink_iz = calculate_electron_sink_due_to_iz_cons(
            solution.views.gauss.solution.conservative,
            solution.atomic_parameters["Eiz"],
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
        _ensure_simple_phys(solution)
        solution.calculate_electron_sink_due_to_rec(which="full")
        _assign_simple_view(solution, "electron_sink_rec", solution.views.glob.sources.electron_sink_rec)
    elif which == "full":
        _ensure_full_solution(solution)
        solution.views.glob.sources.electron_sink_rec = calculate_electron_sink_due_to_rec_cons(
            solution.views.glob.solution.conservative,
            solution.atomic_parameters["Erec"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["charge_scale"],
            solution._cons_idx,
        )
    elif which == "gauss":
        _ensure_gauss_solution(solution)
        solution.views.gauss.sources.electron_sink_rec = calculate_electron_sink_due_to_rec_cons(
            solution.views.gauss.solution.conservative,
            solution.atomic_parameters["Erec"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["charge_scale"],
            solution._cons_idx,
        )


def calculate_electron_gain_due_to_rec(solution, which="simple"):
    _require_atomic_key(solution, "rec", "Please, provide recombination atomic settings for the simulation")
    if which == "simple":
        _ensure_simple_phys(solution)
        solution.calculate_electron_gain_due_to_rec(which="full")
        _assign_simple_view(solution, "electron_gain_rec", solution.views.glob.sources.electron_gain_rec)
    elif which == "full":
        _ensure_full_solution(solution)
        solution.views.glob.sources.electron_gain_rec = calculate_electron_gain_due_to_rec_cons(
            solution.views.glob.solution.conservative,
            solution.atomic_parameters["rec"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["charge_scale"],
            solution._cons_idx,
        )
    elif which == "gauss":
        _ensure_gauss_solution(solution)
        solution.views.gauss.sources.electron_gain_rec = calculate_electron_gain_due_to_rec_cons(
            solution.views.gauss.solution.conservative,
            solution.atomic_parameters["rec"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["charge_scale"],
            solution._cons_idx,
        )


def calculate_electron_sink_due_to_cooling_factor(solution, which="simple"):
    _require_atomic_key(
        solution,
        "cooling_factor",
        "Please, provide atomic settings for electron losses due to cooling factor for the simulation",
    )
    if which == "simple":
        _ensure_simple_phys(solution)
        solution.calculate_electron_sink_due_to_cooling_factor(which="full")
        _assign_simple_view(solution, "electron_sink_cooling_factor", solution.views.glob.sources.electron_sink_cooling_factor)
    elif which == "full":
        _ensure_full_solution(solution)
        solution.views.glob.sources.electron_sink_cooling_factor = calculate_electron_sink_due_to_cooling_factor_cons(
            solution.views.glob.solution.conservative,
            solution.atomic_parameters["cooling_factor"],
            solution.parameters["physics"]["impurity_concentration"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["charge_scale"],
        )
    elif which == "gauss":
        _ensure_gauss_solution(solution)
        solution.views.gauss.sources.electron_sink_cooling_factor = calculate_electron_sink_due_to_cooling_factor_cons(
            solution.views.gauss.solution.conservative,
            solution.atomic_parameters["cooling_factor"],
            solution.parameters["physics"]["impurity_concentration"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["charge_scale"],
        )


def calculate_cooling_factor(solution, which="simple"):
    _require_atomic_key(solution, "cooling_factor", "Please, provide atomic settings for the cooling factor for the simulation")
    if which == "simple":
        _ensure_simple_phys(solution)
        solution.calculate_cooling_factor(which="full")
        _assign_simple_view(solution, "cooling_factor", solution.views.glob.sources.cooling_factor)
    elif which == "full":
        _ensure_full_solution(solution)
        solution.views.glob.sources.cooling_factor = calculate_cooling_factor_cons(
            solution.views.glob.solution.conservative,
            solution.atomic_parameters["cooling_factor"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["physics"]["Mref"],
            solution.parameters["adimensionalization"]["charge_scale"],
        )
    elif which == "gauss":
        _ensure_gauss_solution(solution)
        solution.views.gauss.sources.cooling_factor = calculate_cooling_factor_cons(
            solution.views.gauss.solution.conservative,
            solution.atomic_parameters["cooling_factor"],
            solution.parameters["physics"]["impurity_concentration"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
        )


def calculate_cx_source(solution, which="simple"):
    _require_atomic_key(solution, "iz", "Please, provide ionization atomic settings for the simulation")
    if which == "simple":
        _ensure_simple_phys(solution)
        solution.calculate_cx_source(which="full")
        _assign_simple_view(solution, "cx_source", solution.views.glob.sources.cx_source)
    elif which == "full":
        _ensure_full_solution(solution)
        solution.views.glob.sources.cx_source = calculate_cx_source_cons(
            solution.views.glob.solution.conservative,
            solution.atomic_parameters["cx"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution._cons_idx,
        )
    elif which == "gauss":
        _ensure_gauss_solution(solution)
        solution.views.gauss.sources.cx_source = calculate_cx_source_cons(
            solution.views.gauss.solution.conservative,
            solution.atomic_parameters["cx"],
            solution.parameters["adimensionalization"]["temperature_scale"],
            solution.parameters["adimensionalization"]["density_scale"],
            solution.parameters["physics"]["Mref"],
            solution._cons_idx,
        )


def _require_atomic_key(solution, key, message):
    if solution.atomic_parameters is None:
        raise ValueError("Please, provide atomic settings for the simulation")
    if key not in solution.atomic_parameters.keys():
        raise ValueError(message)


def _ensure_simple_phys(solution):
    if not solution._simple_phys_initialized:
        print("Initializing physical solution first")
        solution.init_phys_variables("simple")


def _ensure_full_solution(solution):
    if not solution._combined_to_full:
        solution.recombine_full_solution()


def _ensure_gauss_solution(solution):
    if solution.views.gauss.solution.conservative is None:
        print("Initializing values in gauss points first")
        solution.calculate_in_gauss_points()


def _assign_simple_view(solution, field_name, full_values):
    simple_values = np.zeros(solution.mesh.vertices_glob.shape[0])
    simple_values[solution.mesh.connectivity_glob.reshape(-1, 1).ravel()] = full_values.reshape(
        solution.views.glob.solution.conservative.shape[0] * solution.views.glob.solution.conservative.shape[1]
    )
    setattr(solution.views.simple.sources, field_name, simple_values)
