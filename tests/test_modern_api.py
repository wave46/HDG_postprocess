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
    solution.mesh.metadata.reference_element = _load_reference_element(cfg["reference_element"])
    if cfg.get("with_atomic_setup"):
        solution.additional_parameters.set_atomic(generate_baselines._make_atomic_params(cfg["radiation_model"]))
        solution.additional_parameters.set_neutral_diffusion(
            generate_baselines._make_dnn_params(),
            solution.parameters["adimensionalization"],
        )
        solution.parameters["physics"]["R_E"] = cfg["r_e_override"]

    full_cons = solution.fields.conservative(view="full")
    simple_phys = solution.fields.physical(view="simple")
    gauss_phys = solution.fields.physical(view="gauss")
    gauss_grad_phys = solution.fields.physical(view="gauss", gradients=True)
    boundary_view = solution.assembly.boundary()
    boundary_gauss_view = solution.assembly.boundary_gauss()
    boundary_cons = solution.fields.conservative(view="boundary")
    boundary_skeleton = solution.fields.conservative(view="boundary", skeleton=True)
    boundary_gauss_cons = solution.fields.conservative(view="boundary_gauss")
    boundary_gauss_skeleton = solution.fields.conservative(view="boundary_gauss", skeleton=True)
    boundary_gauss_cons_cached = solution.fields.conservative(view="boundary_gauss")
    boundary_gauss_skeleton_cached = solution.fields.conservative(view="boundary_gauss", skeleton=True)
    axis = solution.equilibrium.define_axis()
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
    assert gauss_phys.shape[:2] == gauss_grad_phys.shape[:2]
    assert gauss_grad_phys.shape[-2] == baseline["metadata"]["nphys"]
    assert solution.views.glob.solution.conservative is full_cons
    assert solution.views.simple.solution.physical is simple_phys
    assert solution.views.gauss.solution.physical is gauss_phys
    assert solution.views.gauss.gradient.physical is gauss_grad_phys
    assert boundary_view is solution.views.boundary
    assert boundary_gauss_view is solution.views.boundary_gauss
    assert boundary_cons is solution.views.boundary.solution.conservative
    assert boundary_skeleton is solution.views.boundary.solution_skeleton.conservative
    assert boundary_gauss_cons is solution.views.boundary_gauss.solution.conservative
    assert boundary_gauss_skeleton is solution.views.boundary_gauss.solution_skeleton.conservative
    assert boundary_gauss_cons_cached is boundary_gauss_cons
    assert boundary_gauss_skeleton_cached is boundary_gauss_skeleton
    assert axis is solution.summary.equilibrium.axis
    assert solution.metadata.flags.combined_gauss
    assert solution.metadata.flags.combined_boundary
    assert solution.metadata.flags.combined_boundary_gauss
    assert solution.metadata.flags.gauss_phys_initialized
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
    mesh.assembly.full()
    connectivity_big = mesh.geometry.connectivity_big

    assert mesh.metadata.p_order == baseline["p_order"]
    assert mesh.global_state.n_elements == baseline["nelems_glob"]
    assert connectivity_big.shape[0] == baseline["connectivity_big_shape"][0]
