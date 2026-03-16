import numpy as np
import scipy.io

from hdg_postprocess.api import load_mesh, load_solution

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


def test_modern_solution_api(manifest_path, baselines_dir):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["power_balance_with_cooling"]
    baseline = load_baseline(baselines_dir, "power_balance_with_cooling")

    solution = load_solution(
        cfg["solution_path"],
        cfg["solution_base"],
        cfg.get("mesh_path"),
        cfg.get("mesh_base"),
        cfg["n_partitions"],
    )
    solution.legacy.mesh.reference_element = _load_reference_element(cfg["reference_element"])
    if cfg.get("with_atomic_setup"):
        solution.legacy.atomic_parameters = generate_baselines._make_atomic_params(cfg["radiation_model"])
        solution.legacy.dnn_parameters = generate_baselines._make_dnn_params()
        solution.legacy.parameters["physics"]["R_E"] = cfg["r_e_override"]

    full_cons = solution.fields.conservative(view="full")
    simple_phys = solution.fields.physical(view="simple")
    profile = solution.sample.line(
        baseline["sampled_profile"]["r"],
        baseline["sampled_profile"]["z"],
        cfg["profile_variables"],
    )
    midplane_point = baseline["interpolated_points"]["points"]["midplane"][0]
    point = solution.sample.point(midplane_point[0], midplane_point[1], ["n", "te", "ti"])
    power_balance = solution.analysis.power_balance()
    boundary_summary = solution.analysis.boundary_summary()

    assert full_cons.shape[0] == baseline["mesh"]["nelems_glob"]
    assert simple_phys.shape[1] == baseline["metadata"]["nphys"]
    assert solution.views.glob.solution.conservative is full_cons
    assert solution.views.simple.solution.physical is simple_phys
    assert solution.summary.boundary.profile is boundary_summary
    assert set(point.keys()) == {"n", "te", "ti"}
    assert "k" not in profile or len(profile["k"]) == len(baseline["sampled_profile"]["r"])
    assert np.isclose(power_balance["total_loss"], baseline["power_balance"]["total_loss"])
    assert np.allclose(boundary_summary["b_n"][:10], baseline["boundary_summary"]["bn_head"])


def test_modern_mesh_api(manifest_path, baselines_dir):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["legacy_mesh_west"]
    baseline = load_baseline(baselines_dir, "legacy_mesh_west")

    mesh = load_mesh(cfg["mesh_path"], cfg["mesh_base"], cfg["n_partitions"])
    mesh.topology.recombine_full()
    connectivity_big = mesh.topology.create_big_connectivity()

    assert mesh.metadata["p_order"] == baseline["p_order"]
    assert mesh.legacy.nelems_glob == baseline["nelems_glob"]
    assert connectivity_big.shape[0] == baseline["connectivity_big_shape"][0]
