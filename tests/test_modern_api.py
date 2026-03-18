import numpy as np

from hdg_postprocess.api import (
    configure_solution_setup,
    load_mesh,
    load_reference_element,
    load_solution,
    make_atomic_parameters,
    make_neutral_diffusion_parameters,
    make_turbulence_parameters,
)

from helpers import generate_baselines, load_baseline, require_scenario_data, scenario_map


def test_modern_solution_api(manifest_path, baselines_dir):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["power_balance_with_cooling"]
    require_scenario_data(cfg)
    baseline = load_baseline(baselines_dir, "power_balance_with_cooling")

    solution = load_solution(
        cfg["solution_path"],
        cfg["solution_base"],
        cfg.get("mesh_path"),
        cfg.get("mesh_base"),
        cfg["n_partitions"],
    )
    solution.mesh.metadata.reference_element = load_reference_element(cfg["reference_element"])
    if cfg.get("with_atomic_setup"):
        solution.additional_parameters.set_atomic(
            make_atomic_parameters(cfg["radiation_model"], data_dir="demos/data/atomic")
        )
        solution.additional_parameters.set_neutral_diffusion(make_neutral_diffusion_parameters(), solution.parameters["adimensionalization"])
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
    boundary_flags = np.unique(solution.raw.boundary_infos[0]["boundary_flags"])
    nearest_boundary_idx = solution.mesh.boundary.nearest_face_index(
        boundary_summary["r"][0],
        boundary_summary["z"][0],
        raw_boundary_info=solution.raw.boundary_infos[0],
        boundaries=boundary_flags,
    )

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
    assert nearest_boundary_idx >= 0
    assert set(point.keys()) == {"n", "te", "ti"}
    assert "k" not in profile or len(profile["k"]) == len(baseline["sampled_profile"]["r"])
    assert np.isclose(power_balance["total_loss"], baseline["power_balance"]["total_loss"])
    assert np.allclose(boundary_summary["b_n"][:10], baseline["boundary_summary"]["bn_head"])


def test_modern_mesh_api(manifest_path, baselines_dir):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["legacy_mesh_west"]
    require_scenario_data(cfg)
    baseline = load_baseline(baselines_dir, "legacy_mesh_west")

    mesh = load_mesh(cfg["mesh_path"], cfg["mesh_base"], cfg["n_partitions"])
    mesh.assembly.full()
    connectivity_big = mesh.geometry.connectivity_big

    assert mesh.metadata.p_order == baseline["p_order"]
    assert mesh.global_state.n_elements == baseline["nelems_glob"]
    assert connectivity_big.shape[0] == baseline["connectivity_big_shape"][0]


def test_setup_helpers_compose_with_solution_api(manifest_path):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["embedded_k_model"]
    require_scenario_data(cfg)

    solution = load_solution(
        cfg["solution_path"],
        cfg["solution_base"],
        cfg.get("mesh_path"),
        cfg.get("mesh_base"),
        cfg["n_partitions"],
    )
    configure_solution_setup(
        solution,
        reference_element=cfg["reference_element"],
        turbulence=make_turbulence_parameters(),
    )

    assert solution.mesh.metadata.reference_element is not None
    assert solution.additional_parameters.turbulence["dk_max"] == 1e2
