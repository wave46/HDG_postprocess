import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_GENERATE_BASELINES_PATH = ROOT / "scripts" / "generate_baselines.py"
_SPEC = importlib.util.spec_from_file_location("generate_baselines", _GENERATE_BASELINES_PATH)
generate_baselines = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(generate_baselines)


def load_manifest(manifest_path):
    return json.loads(Path(manifest_path).read_text())


def scenario_map(manifest_path):
    manifest = load_manifest(manifest_path)
    return {scenario["id"]: scenario for scenario in manifest["scenarios"]}


def load_baseline(baselines_dir, scenario_id):
    path = Path(baselines_dir) / f"{scenario_id}.json"
    return json.loads(path.read_text())


def _partition_file_paths(path, name_base, n_partitions):
    base_path = ROOT / path
    if n_partitions == 1:
        return [base_path / f"{name_base}.h5"]
    return [base_path / f"{name_base}_{idx}_{n_partitions}.h5" for idx in range(1, n_partitions + 1)]


def scenario_required_paths(scenario_config):
    paths = []
    n_partitions = scenario_config["n_partitions"]

    if "solution_path" in scenario_config and "solution_base" in scenario_config:
        paths.extend(_partition_file_paths(scenario_config["solution_path"], scenario_config["solution_base"], n_partitions))

    if "mesh_path" in scenario_config and "mesh_base" in scenario_config:
        mesh_partitions = n_partitions if scenario_config["kind"] == "mesh" else scenario_config.get("n_partitions", 1)
        paths.extend(_partition_file_paths(scenario_config["mesh_path"], scenario_config["mesh_base"], mesh_partitions))

    if "reference_element" in scenario_config:
        paths.append(ROOT / scenario_config["reference_element"])

    if scenario_config.get("with_atomic_setup"):
        atomic_dir = ROOT / "demos" / "data" / "atomic"
        atomic_files = [
            "alpha_iz.npy",
            "alpha_rec_2.1.8JH.npy",
            "alpha_energy_iz.npy",
            "alpha_energy_rec.npy",
        ]
        if scenario_config.get("radiation_model") == "nitrogen_cooling":
            atomic_files.append("LZ_Nitrogen_adas_fit_te_2e-1_4e3.npy")
        elif scenario_config.get("radiation_model") == "tungsten_cooling":
            atomic_files.append("LZ_Tungsten_adas_fit_te_2e0_4e4.npy")
        paths.extend(atomic_dir / name for name in atomic_files)

    return paths


def missing_scenario_paths(scenario_config):
    return [path for path in scenario_required_paths(scenario_config) if not path.exists()]


def require_scenario_data(scenario_config):
    missing = missing_scenario_paths(scenario_config)
    if missing:
        shown = ", ".join(str(path.relative_to(ROOT)) for path in missing[:3])
        if len(missing) > 3:
            shown += f", ... (+{len(missing) - 3} more)"
        pytest.skip(f"Scenario '{scenario_config['id']}' requires local demo data not present in this checkout: {shown}")


def build_current_baseline(scenario_config):
    require_scenario_data(scenario_config)
    if scenario_config["kind"] == "solution":
        return generate_baselines._collect_solution_baseline(scenario_config)
    if scenario_config["kind"] == "mesh":
        return generate_baselines._collect_mesh_baseline(scenario_config)
    raise ValueError(f"Unsupported scenario kind: {scenario_config['kind']}")


def assert_allclose_nested(current, expected, *, rtol=1e-9, atol=1e-9, path="root"):
    if isinstance(expected, dict):
        assert isinstance(current, dict), f"{path}: expected dict, got {type(current)}"
        assert set(current.keys()) == set(expected.keys()), (
            f"{path}: key mismatch current={sorted(current.keys())} expected={sorted(expected.keys())}"
        )
        for key in expected:
            assert_allclose_nested(current[key], expected[key], rtol=rtol, atol=atol, path=f"{path}.{key}")
        return

    if isinstance(expected, list):
        assert isinstance(current, (list, tuple)), f"{path}: expected list-like, got {type(current)}"
        current_items = list(current)
        assert len(current_items) == len(expected), f"{path}: length mismatch {len(current_items)} != {len(expected)}"
        for idx, (cur_item, exp_item) in enumerate(zip(current_items, expected)):
            assert_allclose_nested(cur_item, exp_item, rtol=rtol, atol=atol, path=f"{path}[{idx}]")
        return

    if isinstance(expected, float):
        assert np.isclose(current, expected, rtol=rtol, atol=atol, equal_nan=True), (
            f"{path}: {current} != {expected} within rtol={rtol}, atol={atol}"
        )
        return

    assert current == expected, f"{path}: {current!r} != {expected!r}"
