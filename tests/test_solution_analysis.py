import numpy as np
import scipy.io
from pathlib import Path

from hdg_postprocess.formats import load_from_file

from helpers import generate_baselines, load_baseline, require_scenario_data, scenario_map


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
    require_scenario_data(cfg)
    baseline = load_baseline(baselines_dir, "power_balance_with_cooling")

    sol = load_from_file.load_HDG_solution_from_file(
        cfg["solution_path"],
        cfg["solution_base"],
        cfg.get("mesh_path"),
        cfg.get("mesh_base"),
        cfg["n_partitions"],
    )
    sol.mesh.metadata.reference_element = _load_reference_element(cfg["reference_element"])
    sol.additional_parameters.set_atomic(generate_baselines._make_atomic_params(cfg["radiation_model"]))
    sol.additional_parameters.set_neutral_diffusion(
        generate_baselines._make_dnn_params(),
        sol.parameters["adimensionalization"],
    )
    sol.parameters["physics"]["R_E"] = cfg["r_e_override"]

    boundary_summary = sol.analysis.boundary_summary()
    power_balance = sol.analysis.power_balance()

    assert np.allclose(boundary_summary["b_n"][:10], baseline["boundary_summary"]["bn_head"])
    assert np.isclose(np.sum(boundary_summary["ds"]), baseline["boundary_summary"]["ds_total"])
    assert np.isclose(power_balance["total_loss"], baseline["power_balance"]["total_loss"])
    assert np.isclose(power_balance["total_wall_loss"], baseline["power_balance"]["total_wall_loss"])


def test_boundary_summary_subset(manifest_path):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["power_balance_with_cooling"]
    require_scenario_data(cfg)

    sol = load_from_file.load_HDG_solution_from_file(
        cfg["solution_path"],
        cfg["solution_base"],
        cfg.get("mesh_path"),
        cfg.get("mesh_base"),
        cfg["n_partitions"],
    )
    sol.mesh.metadata.reference_element = _load_reference_element(cfg["reference_element"])
    sol.additional_parameters.set_atomic(generate_baselines._make_atomic_params(cfg["radiation_model"]))
    sol.additional_parameters.set_neutral_diffusion(
        generate_baselines._make_dnn_params(),
        sol.parameters["adimensionalization"],
    )
    sol.parameters["physics"]["R_E"] = cfg["r_e_override"]

    default_summary = sol.analysis.boundary_summary()

    boundary_ids = np.unique(sol.raw.boundary_infos[0]["boundary_flags"])
    subset_summary = sol.analysis.boundary_summary(
        boundaries=[int(boundary_ids[0])],
        variables=["b_n", "neutral_flux"],
    )

    assert set(subset_summary.keys()) == {"time", "r", "z", "psi", "dl", "ds", "b_n", "neutral_flux"}
    assert subset_summary["b_n"].shape[0] < default_summary["b_n"].shape[0]


def test_boundary_summary_neutral_wall_balance():
    root = Path(__file__).resolve().parents[1]
    solution_dir = root / "demos" / "data" / "solutions" / "limiter_case" / "diffred_test_neutralsgammapressure"
    solution_base = "Sol2D_WEST_60527_P8_DPe0.100E+02_DPai0.314E+06_DPae0.105E+08"
    reference_element = root / "demos" / "data" / "reference_elements" / "reference_triangle_P8.mat"
    atomic_dir = root / "demos" / "data" / "atomic"

    required = [
        solution_dir / f"{solution_base}.h5",
        reference_element,
        atomic_dir / "alpha_iz.npy",
        atomic_dir / "alpha_rec_2.1.8JH.npy",
        atomic_dir / "alpha_energy_iz.npy",
        atomic_dir / "alpha_energy_rec.npy",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        shown = ", ".join(str(path.relative_to(root)) for path in missing[:3])
        if len(missing) > 3:
            shown += f", ... (+{len(missing) - 3} more)"
        import pytest

        pytest.skip(f"Neutral wall diagnostic test requires local demo data not present in this checkout: {shown}")

    sol = load_from_file.load_HDG_solution_from_file(
        f"{solution_dir}/",
        solution_base,
        n_partitions=1,
    )
    sol.mesh.metadata.reference_element = _load_reference_element(reference_element)
    sol.additional_parameters.set_atomic(generate_baselines._make_atomic_params("none"))

    summary = sol.analysis.boundary_summary(
        boundaries=[5, 6, 9],
        variables=[
            "boundary_flag",
            "boundary_condition_code",
            "gamma_parallel_wall",
            "gamma_perp_wall",
            "gamma_pinch_wall",
            "gamma_puff_wall",
            "gamma_pump_wall",
            "neutral_diff_flux",
            "neutral_pgrad_flux",
            "neutral_conv_flux",
            "neutral_flux",
            "neutral_numerical_flux",
            "neutral_wall_balance",
        ],
    )

    assert set(np.unique(summary["boundary_flag"])) <= {5, 6, 9}
    assert set(np.unique(summary["boundary_condition_code"])) <= {50, 55, 56}
    assert np.allclose(
        summary["neutral_flux"],
        summary["neutral_diff_flux"] + summary["neutral_pgrad_flux"] + summary["neutral_conv_flux"],
    )
    assert np.allclose(
        summary["neutral_wall_balance"],
        summary["gamma_parallel_wall"]
        - summary["gamma_perp_wall"]
        - summary["gamma_pinch_wall"]
        - summary["neutral_flux"]
        + summary["gamma_puff_wall"]
        - summary["gamma_pump_wall"]
        + summary["neutral_numerical_flux"],
    )

    for key in (
        "gamma_parallel_wall",
        "gamma_perp_wall",
        "gamma_pinch_wall",
        "gamma_puff_wall",
        "gamma_pump_wall",
        "neutral_diff_flux",
        "neutral_pgrad_flux",
        "neutral_conv_flux",
        "neutral_flux",
        "neutral_numerical_flux",
        "neutral_wall_balance",
    ):
        assert np.all(np.isfinite(summary[key]))
