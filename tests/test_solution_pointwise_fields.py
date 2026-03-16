import json

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


def _load_solution(cfg):
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
    sol.define_interpolators()
    return sol


def test_pointwise_accessors_match_embedded_k_baseline(manifest_path):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["embedded_k_model"]
    baseline = json.load(open("tests/baselines/embedded_k_model.json"))

    sol = _load_solution(cfg)
    sol.additional_parameters.set_turbulence(
        {"dk_min": 1e-6, "dk_max": 1e2, "dk_min_adim": 0.0, "dk_max_adim": 0.0},
        sol.parameters["adimensionalization"],
    )

    point = baseline["interpolated_points"]["values"]["midplane"][0]
    r = point["r"]
    z = point["z"]

    assert np.isclose(sol.n(r, z)[0], point["n"][0])
    assert np.isclose(sol.ti(r, z)[0], point["ti"][0])
    assert np.isclose(sol.te(r, z)[0], point["te"][0])
    assert np.isclose(sol.nn(r, z)[0], point["nn"][0])
    assert np.isclose(sol.M(r, z)[0], point["M"][0])
    assert np.isclose(sol.k(r, z)[0], point["k"][0])
    assert np.isfinite(sol.dk(r, z)[0])
    assert np.isfinite(sol.grad_ti(r, z, "x"))
    assert np.isfinite(sol.psi(r, z))


def test_pointwise_source_and_field_accessors_are_finite(manifest_path):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["power_balance_with_cooling"]
    baseline = json.load(open("tests/baselines/power_balance_with_cooling.json"))

    sol = _load_solution(cfg)
    sol.additional_parameters.set_atomic(generate_baselines._make_atomic_params(cfg["radiation_model"]))
    sol.additional_parameters.set_neutral_diffusion(
        generate_baselines._make_dnn_params(),
        sol.parameters["adimensionalization"],
    )
    sol.parameters["physics"]["R_E"] = cfg["r_e_override"]

    point = baseline["interpolated_points"]["values"]["midplane"][0]
    r = point["r"]
    z = point["z"]

    assert np.isfinite(sol.B(r, z, "theta"))
    assert np.isfinite(sol.grad_B(r, z, "theta", "x"))
    assert np.isfinite(sol.Q_e_loss_iz(r, z))
    assert np.isfinite(sol.Q_e_loss_rec(r, z))
    assert np.isfinite(sol.Q_e_gain_rec(r, z))
    assert np.isfinite(sol.Q_e_loss_tot(r, z))
    assert np.isfinite(sol.Q_i_gain_iz(r, z))
    assert np.isfinite(sol.Q_i_loss_tot(r, z))
    assert np.isfinite(sol.Q_loss_tot(r, z))
