import pytest

from helpers import assert_allclose_nested, build_current_baseline, load_baseline, scenario_map


SCENARIO_IDS = [
    "legacy_first",
    "legacy_mesh_west",
    "embedded_k_model",
    "power_balance_no_heating",
    "power_balance_with_cooling",
]


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_scenario_matches_frozen_baseline(manifest_path, baselines_dir, scenario_id):
    scenarios = scenario_map(manifest_path)
    current = build_current_baseline(scenarios[scenario_id])
    expected = load_baseline(baselines_dir, scenario_id)
    assert_allclose_nested(current, expected)


@pytest.mark.parametrize(
    ("scenario_id", "expected_format", "expected_partitions"),
    [
        ("legacy_first", "legacy_partitioned_external_mesh", 8),
        ("legacy_mesh_west", "partitioned_external_mesh", 8),
        ("embedded_k_model", "embedded_mesh_single_file", 1),
        ("power_balance_no_heating", "embedded_mesh_single_file", 1),
        ("power_balance_with_cooling", "embedded_mesh_single_file", 1),
    ],
)
def test_manifest_covers_expected_dataset_modes(manifest_path, scenario_id, expected_format, expected_partitions):
    scenarios = scenario_map(manifest_path)
    scenario = scenarios[scenario_id]
    assert scenario["format_family"] == expected_format
    assert scenario["n_partitions"] == expected_partitions


def test_k_model_baseline_contains_k_in_profile_and_points(baselines_dir):
    baseline = load_baseline(baselines_dir, "embedded_k_model")
    assert "k" in baseline["sampled_profile"]["values"]

    point_groups = baseline["interpolated_points"]["values"]
    flat_points = point_groups["midplane"] + point_groups["off_midplane"]
    assert flat_points, "expected at least one interpolated point"
    assert all("k" in point for point in flat_points)


@pytest.mark.parametrize("scenario_id", ["power_balance_no_heating", "power_balance_with_cooling"])
def test_power_balance_scenarios_include_boundary_and_power_outputs(baselines_dir, scenario_id):
    baseline = load_baseline(baselines_dir, scenario_id)
    assert "power_balance" in baseline
    assert "boundary_summary" in baseline
    assert "total_loss" in baseline["power_balance"]
    assert "ds_total" in baseline["boundary_summary"]
