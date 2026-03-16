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


def test_grouped_caches_sync_physical_and_equilibrium(manifest_path):
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

    assert sol._grouped_caches["physical"]["simple"]["solution"] is sol.solution_simple_phys
    assert sol._grouped_caches["physical"]["glob"]["solution"] is sol.solution_glob_phys
    assert sol._grouped_caches["equilibrium"]["axis"]["r"] == sol.r_axis
    assert sol._grouped_caches["equilibrium"]["glob"]["qcyl"] is sol.qcyl_glob
    assert sol._summary.equilibrium.axis.r == sol.r_axis
    assert sol._summary.equilibrium.axis.z == sol.z_axis
    assert sol._summary.equilibrium.glob.a is sol.a_glob
    assert sol._summary.equilibrium.simple.a is sol.a_simple
    assert sol._summary.equilibrium.glob.qcyl is sol.qcyl_glob
    assert sol._summary.equilibrium.simple.qcyl is sol.qcyl_simple
    assert sol._grouped_caches["derived"]["simple"]["dk"] is sol.dk_simple
    assert sol._grouped_caches["derived"]["glob"]["dk"] is sol.dk_glob
    assert sol._views.simple.solution.physical is sol.solution_simple_phys
    assert sol._views.glob.solution.conservative is sol.solution_glob
    assert sol._views.glob.gradient.physical is sol.gradient_glob_phys
    assert sol._views.simple.derived.dk is sol.dk_simple
    assert sol._views.glob.derived.dk is sol.dk_glob


def test_grouped_caches_sync_neutral_derived_fields(manifest_path):
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

    assert sol._grouped_caches["derived"]["simple"]["dnn"] is sol.dnn_simple
    assert sol._grouped_caches["derived"]["simple"]["dnn_with_nn_collision"] is sol.dnn_simple_with_nn_collision_simple
    assert sol._grouped_caches["derived"]["glob"]["dnn_with_nn_collision_legacy"] is sol.dnn_simple_with_nn_collision
    assert sol._grouped_caches["derived"]["simple"]["mfp"] is sol.mfp_simple
    assert sol._views.simple.derived.dnn is sol.dnn_simple
    assert sol._views.simple.derived.dnn_with_nn_collision is sol.dnn_simple_with_nn_collision_simple
    assert sol._views.simple.derived.mfp is sol.mfp_simple
    assert sol._views.glob.derived.dnn_with_nn_collision is sol.dnn_simple_with_nn_collision


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

    assert sol._parameter_state.atomic is sol.atomic_parameters
    assert sol._parameter_state.neutral_diffusion is sol.dnn_parameters
    assert sol._parameter_state.turbulence is sol.dk_parameters
    assert sol._parameter_state.neutral_diffusion["dnn_max_adim"] == sol.dnn_parameters["dnn_max_adim"]
    assert sol._parameter_state.turbulence["dk_max_adim"] == sol.dk_parameters["dk_max_adim"]
    assert sol._atomic_rates.ionization_simple is sol.ionization_rate_simple
    assert sol._atomic_rates.recombination_simple is sol.recombination_rate_simple
    assert sol._atomic_rates.cx_simple is sol.cx_rate_simple
    assert sol._interpolators_state.sample is sol.sample_interpolator
    assert sol._interpolators_state.solution is sol.solution_interpolators
    assert sol._interpolators_state.gradient is sol.gradient_interpolators
    assert sol._interpolators_state.field is sol.field_interpolators
    assert sol._interpolators_state.qcyl is sol.qcyl_interpolator


def test_grouped_caches_sync_sources_and_totals(manifest_path):
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

    assert sol._grouped_caches["sources"]["ionization_source"]["simple"] is sol.ionization_source_simple
    assert sol._grouped_caches["sources"]["electron_sink_rec"]["simple"] is sol.electron_sink_rec_simple
    assert sol._grouped_caches["sources"]["cx_source"]["simple"] is sol.cx_source_simple
    assert sol._grouped_caches["sources"]["ohmic_source"]["gauss"] is sol.ohmic_source_gauss
    assert sol._grouped_caches["sources"]["ion_gain_iz"]["total"] == sol.ion_gain_iz_total
    assert sol._grouped_caches["sources"]["electron_sink_iz"]["total"] == sol.electron_sink_iz_total
    assert sol._summary.sources.ion_gain_iz_total == sol.ion_gain_iz_total
    assert sol._summary.sources.electron_sink_iz_total == sol.electron_sink_iz_total
    assert sol._summary.sources.ohmic_source_total == sol.ohmic_source_total
    assert sol._views.glob.sources.ionization_source is sol.ionization_source
    assert sol._views.simple.sources.ionization_source is sol.ionization_source_simple
    assert sol._views.gauss.sources.ionization_source is sol.ionization_source_gauss
    assert sol._views.glob.sources.electron_sink_rec is sol.electron_sink_rec
    assert sol._views.simple.sources.electron_sink_rec is sol.electron_sink_rec_simple
    assert sol._views.gauss.sources.electron_sink_rec is sol.electron_sink_rec_gauss
    assert sol._views.glob.sources.cx_source is sol.cx_source
    assert sol._views.simple.sources.cx_source is sol.cx_source_simple
    assert sol._views.gauss.sources.cx_source is sol.cx_source_gauss
    assert sol._views.glob.sources.ohmic_source is sol.ohmic_source
    assert sol._views.simple.sources.ohmic_source is sol.ohmic_source_simple
    assert sol._views.gauss.sources.ohmic_source is sol.ohmic_source_gauss
    assert sol._summary.boundary.boundary_summary is sol.boundary_summary
    assert sol._summary.boundary.ion_energy_sheath_loss_total == sol.ion_energy_sheath_loss_total
    assert sol._summary.boundary.electron_energy_sheath_loss_total == sol.electron_energy_sheath_loss_total


def test_grouped_caches_sync_representations(manifest_path):
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

    assert sol._grouped_caches["representations"]["simple"]["solution"] is sol.solution_simple
    assert sol._grouped_caches["representations"]["glob"]["solution"] is sol.solution_glob
    assert sol._grouped_caches["representations"]["gauss"]["solution"] is sol.solution_gauss
    assert sol._grouped_caches["representations"]["boundary"]["solution"] is sol.solution_boundary
    assert sol._grouped_caches["representations"]["boundary_gauss"]["solution"] is sol.solution_boundary_gauss
    assert sol._grouped_caches["representations"]["gauss"]["magnetic_field"] is sol.magnetic_field_gauss
    assert sol._views.simple.solution.conservative is sol.solution_simple
    assert sol._views.simple.gradient.conservative is sol.gradient_simple
    assert sol._views.simple.equilibrium.magnetic_field is sol.magnetic_field_simple
    assert sol._views.simple.equilibrium.jtor is sol.jtor_simple
    assert sol._views.simple.equilibrium.poloidal_flux is sol.poloidal_flux_simple
    assert sol._views.glob.equilibrium.magnetic_field is sol.magnetic_field_glob
    assert sol._views.glob.equilibrium.magnetic_field_unit is sol.magnetic_field_unit_glob
    assert sol._views.glob.equilibrium.jtor is sol.jtor_glob
    assert sol._views.glob.equilibrium.poloidal_flux is sol.poloidal_flux_glob
    assert sol._views.gauss.solution.conservative is sol.solution_gauss
    assert sol._views.gauss.gradient.conservative is sol.gradient_gauss
    assert sol._views.gauss.equilibrium.magnetic_field is sol.magnetic_field_gauss
    assert sol._views.gauss.equilibrium.magnetic_field_unit is sol.magnetic_field_unit_gauss
    assert sol._views.gauss.equilibrium.jtor is sol.jtor_gauss
    assert sol._views.gauss.equilibrium.poloidal_flux is sol.poloidal_flux_gauss
    assert sol._views.boundary.solution.conservative is sol.solution_boundary
    assert sol._views.boundary.solution_skeleton.conservative is sol.solution_skeleton_boundary
    assert sol._views.boundary.equilibrium.magnetic_field is sol.magnetic_field_boundary
    assert sol._views.boundary.equilibrium.magnetic_field_unit is sol.magnetic_field_unit_boundary
    assert sol._views.boundary.equilibrium.poloidal_flux is sol.poloidal_flux_boundary
    assert sol._views.boundary_gauss.solution.conservative is sol.solution_boundary_gauss
    assert sol._views.boundary_gauss.solution_skeleton.conservative is sol.solution_skeleton_boundary_gauss
    assert sol._views.boundary_gauss.equilibrium.magnetic_field is sol.magnetic_field_boundary_gauss
    assert sol._views.boundary_gauss.equilibrium.magnetic_field_unit is sol.magnetic_field_unit_boundary_gauss
    assert sol._views.boundary_gauss.equilibrium.poloidal_flux is sol.poloidal_flux_boundary_gauss
