import json
import importlib.util
from pathlib import Path

import numpy as np

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


def build_current_baseline(scenario_config):
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
