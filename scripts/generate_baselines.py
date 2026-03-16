#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io

from hdg_postprocess.formats import load_from_file


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tests" / "baseline_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "tests" / "baselines"


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _load_reference_element(path):
    ref_elem = scipy.io.loadmat(path)
    if "refEl" in ref_elem:
        name = "refEl"
    elif "referenceelement" in ref_elem:
        name = "referenceelement"
    else:
        raise KeyError(f"Unsupported reference-element keys in {path}: {sorted(ref_elem.keys())}")
    ref_dic = {}
    ref_dic["IPcoordinates"] = ref_elem[name][0, 0][0]
    ref_dic["IPweights"] = ref_elem[name][0, 0][1][:, 0]
    ref_dic["N"] = ref_elem[name][0, 0][2]
    ref_dic["Nxi"] = ref_elem[name][0, 0][3]
    ref_dic["Neta"] = ref_elem[name][0, 0][4]
    ref_dic["IPcoordinates1d"] = ref_elem[name][0, 0][5]
    ref_dic["IPweights1d"] = ref_elem[name][0, 0][6]
    ref_dic["N1d"] = ref_elem[name][0, 0][7]
    ref_dic["N1dxi"] = ref_elem[name][0, 0][8]
    ref_dic["faceNodes"] = ref_elem[name][0, 0][9] - 1
    ref_dic["innerNodes"] = ref_elem[name][0, 0][10]
    ref_dic["faceNodes1d"] = ref_elem[name][0, 0][11] - 1
    ref_dic["NodesCoord"] = ref_elem[name][0, 0][12]
    ref_dic["NodesCoord1d"] = ref_elem[name][0, 0][13]
    ref_dic["degree"] = ref_elem[name][0, 0][14]
    return ref_dic


def _make_dnn_params():
    return {
        "const": False,
        "dnn_soft": True,
        "dnn_max": 2e8,
        "dnn_min": 3.0,
        "dnn_w": 0.01,
        "dnn_width": 10,
        "ti_soft": True,
        "ti_min": 1e-6,
        "ti_w": 0.01,
        "ti_width": 10,
    }


def _make_atomic_params(radiation_model):
    atomic_data = ROOT / "demos" / "data" / "atomic"
    atomic = {
        "iz": {
            "database": "AMJUEL 2.1.5JH",
            "alpha": np.load(atomic_data / "alpha_iz.npy"),
            "te_min": 0.1,
            "te_max": 2e4,
            "ne_min": 1e14,
            "ne_max": 1e22,
        },
        "cx": {
            "database": "OpenADAS expanded",
            "alpha": np.array(
                [-1.87744894e01, 4.51800000e-01, -3.58100000e-02, 8.00400000e-03, -6.83700000e-04]
            ),
            "te_min": 0.1,
            "te_max": 2e4,
        },
        "rec": {
            "database": "AMJUEL 2.1.8JH",
            "alpha": np.load(atomic_data / "alpha_rec_2.1.8JH.npy"),
            "te_min": 0.1,
            "te_max": 2e4,
            "ne_min": 1e14,
            "ne_max": 1e22,
        },
        "Eiz": {
            "database": "AMJUEL 2.1.5JH",
            "alpha": np.load(atomic_data / "alpha_energy_iz.npy"),
            "te_min": 0.1,
            "te_max": 2e4,
            "ne_min": 1e14,
            "ne_max": 1e22,
        },
        "Erec": {
            "database": "AMJUEL 2.1.8JH",
            "alpha": np.load(atomic_data / "alpha_energy_rec.npy"),
            "te_min": 0.1,
            "te_max": 2e4,
            "ne_min": 1e14,
            "ne_max": 1e22,
        },
    }
    if radiation_model == "nitrogen_cooling":
        cooling = np.load(atomic_data / "LZ_Nitrogen_adas_fit_te_2e-1_4e3.npy")
        cooling[0] -= np.log(1.60217662e-19)
        atomic["cooling_factor"] = {
            "database": "ADAS",
            "alpha": cooling,
            "te_min": 0.1,
            "te_max": 3e3,
        }
    return atomic


def _roundtrip_names(values):
    return [val.decode("utf-8") if isinstance(val, bytes) else str(val) for val in values]


def _mesh_baseline(mesh, element_probe=None):
    if not mesh.metadata.flags.combined_to_full:
        mesh.geometry.recombine_full()
    mesh.geometry.connectivity_big

    baseline = {
        "mesh_extent": _to_builtin(mesh.metadata.extent),
        "p_order": int(mesh.metadata.p_order),
        "nelems_glob": int(mesh.global_state.n_elements),
        "nvertices_glob": int(mesh.global_state.n_vertices),
        "connectivity_shape": list(mesh.global_state.connectivity.shape),
        "connectivity_big_shape": list(mesh.derived_geometry.connectivity_big.shape),
    }

    if element_probe is not None:
        mesh.geometry.element_locator
        baseline["element_probe"] = {
            "point": list(element_probe),
            "element_number": int(mesh.derived_geometry.element_locator(*element_probe)),
        }

    if mesh.metadata.reference_element is not None:
        mesh.geometry.gauss_volumes
        baseline["gauss_volumes"] = {
            "shape": list(mesh.derived_geometry.gauss_volumes.shape),
            "total": float(mesh.derived_geometry.gauss_volumes.sum()),
            "first_element_sum": float(mesh.derived_geometry.gauss_volumes[0].sum()),
        }

    return baseline


def _pick_inside_points(sol, n_points=6):
    z_mid = 0.0 if sol.mesh.metadata.extent["minz"] <= 0.0 <= sol.mesh.metadata.extent["maxz"] else (
        sol.mesh.metadata.extent["minz"] + sol.mesh.metadata.extent["maxz"]
    ) / 2.0
    candidates_r = np.linspace(sol.mesh.metadata.extent["minr"], sol.mesh.metadata.extent["maxr"], 400)
    inside_mid = [(float(r), float(z_mid)) for r in candidates_r if int(sol.mesh.derived_geometry.element_locator(r, z_mid)) != -1]

    z_off = z_mid + 0.15 * (sol.mesh.metadata.extent["maxz"] - sol.mesh.metadata.extent["minz"])
    z_off = min(sol.mesh.metadata.extent["maxz"], max(sol.mesh.metadata.extent["minz"], z_off))
    inside_off = [(float(r), float(z_off)) for r in candidates_r if int(sol.mesh.derived_geometry.element_locator(r, z_off)) != -1]

    mid = inside_mid[:n_points]
    off = inside_off[: max(0, n_points - len(mid))]
    return {
        "midplane": mid,
        "off_midplane": off,
    }


def _collect_interpolated_values(sol, variables):
    sol.sample.define_interpolators()
    points = _pick_inside_points(sol)
    result = {"points": points, "values": {"midplane": [], "off_midplane": []}}
    pointwise_accessors = {
        "n": sol.pointwise.plasma.n,
        "ti": sol.pointwise.plasma.ti,
        "te": sol.pointwise.plasma.te,
        "u": sol.pointwise.plasma.u,
        "cs": sol.pointwise.plasma.cs,
        "M": sol.pointwise.plasma.mach,
        "nn": sol.pointwise.plasma.nn,
        "dnn": sol.pointwise.plasma.dnn,
        "k": sol.pointwise.plasma.k,
        "dk": sol.pointwise.plasma.dk,
        "mfp_nn": sol.pointwise.plasma.mfp_nn,
        "psi": sol.pointwise.fields.psi,
        "ionization_source": sol.pointwise.sources.ionization_source,
        "ionization_rate": sol.pointwise.sources.ionization_rate,
        "cx_rate": sol.pointwise.sources.cx_rate,
        "Q_e_loss_iz": sol.pointwise.sources.Q_e_loss_iz,
        "Q_e_loss_rec": sol.pointwise.sources.Q_e_loss_rec,
        "Q_e_gain_rec": sol.pointwise.sources.Q_e_gain_rec,
        "Q_e_loss_total": sol.pointwise.sources.Q_e_loss_total,
        "Q_i_gain_iz": sol.pointwise.sources.Q_i_gain_iz,
        "Q_i_loss_rec": sol.pointwise.sources.Q_i_loss_rec,
        "Q_i_loss_cx": sol.pointwise.sources.Q_i_loss_cx,
        "Q_i_loss_total": sol.pointwise.sources.Q_i_loss_total,
        "Q_loss_total": sol.pointwise.sources.Q_loss_total,
    }

    for group_name, group_points in points.items():
        for r, z in group_points:
            point_data = {"r": r, "z": z}
            for variable in variables:
                if variable == "grad_ti":
                    point_data["grad_ti_x"] = _to_builtin(sol.pointwise.gradients.ti(r, z, "x"))
                    point_data["grad_ti_y"] = _to_builtin(sol.pointwise.gradients.ti(r, z, "y"))
                    continue
                if variable == "grad_ti_par":
                    point_data["grad_ti_par"] = _to_builtin(sol.pointwise.gradients.ti_parallel(r, z))
                    continue
                accessor = pointwise_accessors[variable]
                value = accessor(r, z)
                point_data[variable] = _to_builtin(value)
            result["values"][group_name].append(point_data)
    return result


def _collect_profile(sol, variables):
    sol.sample.define_interpolators()
    points = _pick_inside_points(sol, n_points=32)["midplane"]
    r_line = np.array([r for r, _ in points], dtype=float)
    z_line = np.array([z for _, z in points], dtype=float)
    profile = sol.sample.line(r_line, z_line, variables)
    return {
        "r": _to_builtin(r_line),
        "z": _to_builtin(z_line),
        "values": {name: _to_builtin(values) for name, values in profile.items()},
    }


def _collect_phys_summary(sol):
    sol.assembly.simple()
    sol.assembly.full()
    sol.fields.initialize_physical("both")

    phys_names = _roundtrip_names(sol.parameters["physics"]["physical_variable_names"])
    simple_phys = sol.views.simple.solution.physical
    full_phys = sol.views.glob.solution.physical
    summary = {}
    for idx, name in enumerate(phys_names):
        summary[name] = {
            "simple_min": float(np.nanmin(simple_phys[:, idx])),
            "simple_max": float(np.nanmax(simple_phys[:, idx])),
            "full_min": float(np.nanmin(full_phys[..., idx])),
            "full_max": float(np.nanmax(full_phys[..., idx])),
        }
    return summary


def _collect_boundary_summary(sol):
    summary = sol.analysis.boundary_summary() or sol.summary.boundary.profile
    return {
        "keys": sorted(summary.keys()),
        "ds_total": float(np.sum(summary["ds"])),
        "neutral_flux_total": float(np.sum(summary["neutral_flux_skeleton"] * summary["ds"])),
        "qi_bc_total": float(np.sum(summary["q_i_tot_dep_bc_skeleton"] * summary["ds"])),
        "qe_bc_total": float(np.sum(summary["q_e_tot_dep_bc_skeleton"] * summary["ds"])),
        "bn_head": _to_builtin(summary["b_n"][:10]),
    }


def _collect_solution_baseline(config):
    sol = load_from_file.load_HDG_solution_from_file(
        config["solution_path"],
        config["solution_base"],
        config.get("mesh_path"),
        config.get("mesh_base"),
        config["n_partitions"],
    )
    sol.mesh.metadata.reference_element = _load_reference_element(ROOT / config["reference_element"])
    sol.mesh.geometry.element_locator

    if config.get("with_atomic_setup"):
        sol.additional_parameters.set_atomic(_make_atomic_params(config["radiation_model"]))
        sol.additional_parameters.set_neutral_diffusion(
            _make_dnn_params(),
            sol.parameters["adimensionalization"],
        )
        sol.parameters["physics"]["R_E"] = config["r_e_override"]

    baseline = {
        "metadata": {
            "neq": int(sol.neq),
            "nphys": int(sol.nphys),
            "ndim": int(sol.ndim),
            "n_partitions": int(sol.n_partitions),
            "conservative_variable_names": _roundtrip_names(sol.parameters["physics"]["conservative_variable_names"]),
            "physical_variable_names": _roundtrip_names(sol.parameters["physics"]["physical_variable_names"]),
            "element_type": sol.mesh.mesh_parameters["element_type"],
            "nodes_per_element": int(sol.mesh.mesh_parameters["nodes_per_element"]),
        },
        "mesh": _mesh_baseline(sol.mesh),
        "physical_fields": _collect_phys_summary(sol),
        "interpolated_points": _collect_interpolated_values(sol, config["point_sample_variables"]),
        "sampled_profile": _collect_profile(sol, config["profile_variables"]),
    }

    if config.get("power_balance"):
        baseline["power_balance"] = _to_builtin(sol.analysis.power_balance())

    if config.get("boundary_summary"):
        baseline["boundary_summary"] = _collect_boundary_summary(sol)

    return baseline


def _collect_mesh_baseline(config):
    mesh = load_from_file.load_HDG_mesh_from_file(
        config["mesh_path"],
        config["mesh_base"],
        config["n_partitions"],
    )
    mesh.metadata.reference_element = _load_reference_element(ROOT / config["reference_element"])
    return _mesh_baseline(mesh, config.get("element_probe"))


def generate_baselines(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text())
    index = {"version": manifest["version"], "scenarios": []}

    for config in manifest["scenarios"]:
        scenario_id = config["id"]
        print(f"Generating baseline for {scenario_id}")
        if config["kind"] == "solution":
            baseline = _collect_solution_baseline(config)
        elif config["kind"] == "mesh":
            baseline = _collect_mesh_baseline(config)
        else:
            raise ValueError(f"Unsupported scenario kind: {config['kind']}")

        baseline_path = output_dir / f"{scenario_id}.json"
        baseline_path.write_text(json.dumps(_to_builtin(baseline), indent=2, sort_keys=True))
        index["scenarios"].append({"id": scenario_id, "path": str(baseline_path.relative_to(ROOT))})

    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    return index_path


def main():
    parser = argparse.ArgumentParser(description="Generate regression baselines from current demo-covered behavior.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated baseline JSON files.",
    )
    args = parser.parse_args()
    index_path = generate_baselines(Path(args.output_dir))
    print(f"Wrote baseline index to {index_path}")


if __name__ == "__main__":
    main()
