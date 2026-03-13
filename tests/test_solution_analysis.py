import numpy as np
import scipy.io

from hdg_postprocess.formats import load_from_file

from helpers import generate_baselines, load_baseline, scenario_map


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


def test_solution_analysis_surface(manifest_path, baselines_dir):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["power_balance_with_cooling"]
    baseline = load_baseline(baselines_dir, "power_balance_with_cooling")

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

    boundary_summary = sol.summary_along_the_wall()
    power_balance = sol.calculate_power_balance()

    assert np.allclose(boundary_summary["b_n"][:10], baseline["boundary_summary"]["bn_head"])
    assert np.isclose(np.sum(boundary_summary["ds"]), baseline["boundary_summary"]["ds_total"])
    assert np.isclose(power_balance["total_loss"], baseline["power_balance"]["total_loss"])
    assert np.isclose(power_balance["total_wall_loss"], baseline["power_balance"]["total_wall_loss"])
