import numpy as np
import scipy.io

from hdg_postprocess.formats import load_from_file

from helpers import generate_baselines, scenario_map


def _load_reference_element(path):
    ref_elem = scipy.io.loadmat(path)
    key = "refEl" if "refEl" in ref_elem else "referenceelement"
    ref_dic = {}
    ref_dic["IPcoordinates"] = ref_elem[key][0, 0][0]
    ref_dic["IPweights"] = ref_elem[key][0, 0][1][:, 0]
    ref_dic["N"] = ref_elem[key][0, 0][2]
    ref_dic["Nxi"] = ref_elem[key][0, 0][3]
    ref_dic["Neta"] = ref_elem[key][0, 0][4]
    ref_dic["IPcoordinates1d"] = ref_elem[key][0, 0][5]
    ref_dic["IPweights1d"] = ref_elem[key][0, 0][6]
    ref_dic["N1d"] = ref_elem[key][0, 0][7]
    ref_dic["N1dxi"] = ref_elem[key][0, 0][8]
    ref_dic["faceNodes"] = ref_elem[key][0, 0][9] - 1
    ref_dic["innerNodes"] = ref_elem[key][0, 0][10]
    ref_dic["faceNodes1d"] = ref_elem[key][0, 0][11] - 1
    ref_dic["NodesCoord"] = ref_elem[key][0, 0][12]
    ref_dic["NodesCoord1d"] = ref_elem[key][0, 0][13]
    ref_dic["degree"] = ref_elem[key][0, 0][14]
    return ref_dic


def test_container_state_sync_physical_and_equilibrium(manifest_path):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["embedded_k_model"]

    sol = load_from_file.load_HDG_solution_from_file(
        cfg["solution_path"],
        cfg["solution_base"],
        cfg.get("mesh_path"),
        cfg.get("mesh_base"),
        cfg["n_partitions"],
    )
    sol.mesh.reference_element = _load_reference_element(cfg["reference_element"])

    sol.recombine_full_solution()
    sol.recombine_simple_full_solution()
    sol.init_phys_variables("both")
    sol.define_magnetic_axis()
    sol.define_minor_radii(which="full")
    sol.define_minor_radii(which="simple")
    sol.define_qcyl(which="full")
    sol.define_qcyl(which="simple")
    sol.dk_parameters = {"dk_min": 1e-6, "dk_max": 1e2, "dk_min_adim": 0.0, "dk_max_adim": 0.0}
    sol.calculate_dk("full")
    sol.calculate_dk("simple")

    assert sol.summary.equilibrium.axis.r is not None
    assert sol.summary.equilibrium.axis.z is not None
    assert sol.views.glob.equilibrium.a is not None
    assert sol.views.simple.equilibrium.a is not None
    assert sol.views.glob.equilibrium.qcyl is not None
    assert sol.views.simple.equilibrium.qcyl is not None
    assert sol.views.simple.solution.physical is not None
    assert sol.views.glob.solution.conservative is not None
    assert sol.views.glob.gradient.physical is not None
    assert sol.views.simple.derived.dk is not None
    assert sol.views.glob.derived.dk is not None


def test_container_state_sync_neutral_derived_fields(manifest_path):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["power_balance_with_cooling"]

    sol = load_from_file.load_HDG_solution_from_file(
        cfg["solution_path"],
        cfg["solution_base"],
        cfg.get("mesh_path"),
        cfg.get("mesh_base"),
        cfg["n_partitions"],
    )
    sol.mesh.reference_element = _load_reference_element(cfg["reference_element"])
    sol.atomic_parameters = generate_baselines._make_atomic_params(cfg["radiation_model"])
    sol.dnn_parameters = generate_baselines._make_dnn_params()
    sol.parameters["physics"]["R_E"] = cfg["r_e_override"]

    sol.recombine_full_solution()
    sol.recombine_simple_full_solution()
    sol.calculate_dnn("simple")
    sol.calculate_dnn_with_nn_collision("simple")
    sol.calculate_mfp("simple")

    assert sol.views.simple.derived.dnn is not None
    assert sol.views.simple.derived.dnn_with_nn_collision is not None
    assert sol.views.simple.derived.mfp is not None
    assert sol.views.glob.derived.dnn_with_nn_collision is not None


def test_aux_state_sync_parameters_rates_and_interpolators(manifest_path):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["power_balance_with_cooling"]

    sol = load_from_file.load_HDG_solution_from_file(
        cfg["solution_path"],
        cfg["solution_base"],
        cfg.get("mesh_path"),
        cfg.get("mesh_base"),
        cfg["n_partitions"],
    )
    sol.mesh.reference_element = _load_reference_element(cfg["reference_element"])

    atomic_parameters = generate_baselines._make_atomic_params(cfg["radiation_model"])
    dnn_parameters = generate_baselines._make_dnn_params()
    dk_parameters = {"dk_min": 1e-6, "dk_max": 1e2, "dk_min_adim": 0.0, "dk_max_adim": 0.0}

    sol.atomic_parameters = atomic_parameters
    sol.dnn_parameters = dnn_parameters
    sol.dk_parameters = dk_parameters

    sol.recombine_full_solution()
    sol.recombine_simple_full_solution()
    sol.calculate_ionization_rate("simple")
    sol.calculate_recombination_rate("simple")
    sol.calculate_cx_rate("simple")
    sol.define_magnetic_axis()
    sol.define_minor_radii(which="full")
    sol.define_qcyl(which="full")
    sol.define_interpolators()

    assert sol.parameter_state.atomic is sol.atomic_parameters
    assert sol.parameter_state.neutral_diffusion is sol.dnn_parameters
    assert sol.parameter_state.turbulence is sol.dk_parameters
    assert sol.parameter_state.neutral_diffusion["dnn_max_adim"] == sol.dnn_parameters["dnn_max_adim"]
    assert sol.parameter_state.turbulence["dk_max_adim"] == sol.dk_parameters["dk_max_adim"]
    assert sol.atomic_rates.ionization_simple is not None
    assert sol.atomic_rates.recombination_simple is not None
    assert sol.atomic_rates.cx_simple is not None
    assert sol.interpolators.sample is not None
    assert sol.interpolators.solution is not None
    assert sol.interpolators.gradient is not None
    assert sol.interpolators.field is not None
    assert sol.interpolators.qcyl is not None


def test_container_state_sync_sources_and_totals(manifest_path):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["power_balance_with_cooling"]

    sol = load_from_file.load_HDG_solution_from_file(
        cfg["solution_path"],
        cfg["solution_base"],
        cfg.get("mesh_path"),
        cfg.get("mesh_base"),
        cfg["n_partitions"],
    )
    sol.mesh.reference_element = _load_reference_element(cfg["reference_element"])
    sol.atomic_parameters = generate_baselines._make_atomic_params(cfg["radiation_model"])
    sol.dnn_parameters = generate_baselines._make_dnn_params()
    sol.parameters["physics"]["R_E"] = cfg["r_e_override"]

    sol.recombine_full_solution()
    sol.recombine_simple_full_solution()
    sol.calculate_ionization_source("simple")
    sol.calculate_electron_sink_due_to_rec("simple")
    sol.calculate_cx_source("simple")
    sol.calculate_power_balance()

    assert sol.summary.sources.ion_gain_iz_total == sol.ion_gain_iz_total
    assert sol.summary.sources.electron_sink_iz_total == sol.electron_sink_iz_total
    assert sol.summary.sources.ohmic_source_total == sol.ohmic_source_total
    assert sol.views.glob.sources.ionization_source is not None
    assert sol.views.simple.sources.ionization_source is not None
    assert sol.views.glob.sources.electron_sink_rec is not None
    assert sol.views.simple.sources.electron_sink_rec is not None
    assert sol.views.gauss.sources.electron_sink_rec is not None
    assert sol.views.glob.sources.cx_source is not None
    assert sol.views.simple.sources.cx_source is not None
    assert sol.views.gauss.sources.ohmic_source is not None
    assert sol.summary.boundary.profile is not None
    assert sol.summary.boundary.ion_energy_sheath_loss_total == sol.ion_energy_sheath_loss_total
    assert sol.summary.boundary.electron_energy_sheath_loss_total == sol.electron_energy_sheath_loss_total


def test_container_state_sync_representations(manifest_path):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["power_balance_with_cooling"]

    sol = load_from_file.load_HDG_solution_from_file(
        cfg["solution_path"],
        cfg["solution_base"],
        cfg.get("mesh_path"),
        cfg.get("mesh_base"),
        cfg["n_partitions"],
    )
    sol.mesh.reference_element = _load_reference_element(cfg["reference_element"])

    sol.recombine_full_solution()
    sol.recombine_simple_full_solution()
    sol.recombine_boundary_solution()
    sol.calculate_in_gauss_points()
    sol.calculate_in_boundary_gauss_points(np.unique(sol.raw_solution_boundary_infos[0]["boundary_flags"]))

    assert sol.views.simple.solution.conservative is not None
    assert sol.views.simple.gradient.conservative is not None
    assert sol.views.simple.equilibrium.magnetic_field is not None
    assert sol.views.simple.equilibrium.jtor is not None
    assert sol.views.simple.equilibrium.poloidal_flux is not None
    assert sol.views.glob.equilibrium.magnetic_field is not None
    assert sol.views.glob.equilibrium.magnetic_field_unit is not None
    assert sol.views.glob.equilibrium.jtor is not None
    assert sol.views.glob.equilibrium.poloidal_flux is not None
    assert sol.views.gauss.solution.conservative is not None
    assert sol.views.gauss.gradient.conservative is not None
    assert sol.views.gauss.equilibrium.magnetic_field is not None
    assert sol.views.gauss.equilibrium.magnetic_field_unit is not None
    assert sol.views.gauss.equilibrium.jtor is not None
    assert sol.views.gauss.equilibrium.poloidal_flux is not None
    assert sol.views.boundary.solution.conservative is not None
    assert sol.views.boundary.solution_skeleton.conservative is not None
    assert sol.views.boundary.gradient.conservative is not None
    assert sol.views.boundary.equilibrium.magnetic_field is not None
    assert sol.views.boundary.equilibrium.magnetic_field_unit is not None
    assert sol.views.boundary.equilibrium.poloidal_flux is not None
    assert sol.views.boundary_gauss.solution.conservative is not None
    assert sol.views.boundary_gauss.solution_skeleton.conservative is not None
    assert sol.views.boundary_gauss.gradient.conservative is not None
    assert sol.views.boundary_gauss.equilibrium.magnetic_field is not None
    assert sol.views.boundary_gauss.equilibrium.magnetic_field_unit is not None
    assert sol.views.boundary_gauss.equilibrium.poloidal_flux is not None
