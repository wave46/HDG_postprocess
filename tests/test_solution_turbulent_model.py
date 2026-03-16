import numpy as np
import scipy.io

from hdg_postprocess.formats import load_from_file

from helpers import load_baseline, scenario_map


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


def test_solution_turbulent_model_surface(manifest_path, baselines_dir):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["embedded_k_model"]
    baseline = load_baseline(baselines_dir, "embedded_k_model")

    sol = load_from_file.load_HDG_solution_from_file(
        cfg["solution_path"],
        cfg["solution_base"],
        cfg.get("mesh_path"),
        cfg.get("mesh_base"),
        cfg["n_partitions"],
    )
    sol.mesh.reference_element = _load_reference_element(cfg["reference_element"])
    sol.additional_parameters.set_turbulence({
        "dk_min": 1e-6,
        "dk_max": 1e2,
        "dk_min_adim": 0.0,
        "dk_max_adim": 0.0,
    }, sol.parameters["adimensionalization"])

    sol.turbulence.dk("simple")

    assert sol.views.simple.derived.dk.shape[0] == baseline["mesh"]["nvertices_glob"]
    assert sol.views.glob.derived.dk.shape[:2] == sol.views.glob.solution.conservative.shape[:2]
    assert np.isfinite(sol.views.simple.derived.dk).all()
