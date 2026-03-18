import json

from hdg_postprocess.io import detect_mesh_format, detect_solution_format, load_mesh_data, load_solution_data
from silx.io.dictdump import h5todict

from helpers import require_scenario_data, scenario_map


def test_detect_solution_and_mesh_formats(manifest_path):
    scenarios = scenario_map(manifest_path)
    require_scenario_data(scenarios["legacy_first"])
    require_scenario_data(scenarios["embedded_k_model"])
    require_scenario_data(scenarios["legacy_mesh_west"])

    legacy_solution = h5todict(
        f"{scenarios['legacy_first']['solution_path']}{scenarios['legacy_first']['solution_base']}_1_8.h5"
    )
    embedded_solution = h5todict(
        f"{scenarios['embedded_k_model']['solution_path']}{scenarios['embedded_k_model']['solution_base']}.h5"
    )
    external_mesh = h5todict(
        f"{scenarios['legacy_mesh_west']['mesh_path']}{scenarios['legacy_mesh_west']['mesh_base']}_1_8.h5"
    )

    assert detect_solution_format(legacy_solution) == "flat_solution"
    assert detect_solution_format(embedded_solution) == "grouped_solution"
    assert detect_mesh_format(legacy_solution) == "flat_mesh"
    assert detect_mesh_format(embedded_solution) == "grouped_mesh"
    assert detect_mesh_format(external_mesh) == "flat_mesh"


def test_normalized_solution_data_matches_manifest_configuration(manifest_path):
    scenarios = scenario_map(manifest_path)
    config = scenarios["power_balance_no_heating"]
    require_scenario_data(config)

    solution_data = load_solution_data(
        config["solution_path"],
        config["solution_base"],
        None,
        None,
        config["n_partitions"],
    )

    assert solution_data.n_partitions == 1
    assert solution_data.mesh_path == config["solution_path"]
    assert solution_data.mesh_name_base == config["solution_base"]
    assert len(solution_data.raw_solutions) == 1
    assert "magnetic_field" in solution_data.raw_equilibriums[0]
    assert "boundary_flags" in solution_data.raw_solution_boundary_infos[0]


def test_normalized_mesh_data_contains_parallel_metadata(manifest_path):
    scenarios = scenario_map(manifest_path)
    config = scenarios["legacy_mesh_west"]
    require_scenario_data(config)

    mesh_data = load_mesh_data(
        config["mesh_path"],
        config["mesh_base"],
        config["n_partitions"],
    )

    assert mesh_data.n_partitions == 8
    assert mesh_data.mesh_parameters["element_type"] == "triangle"
    assert len(mesh_data.raw_vertices) == 8
    assert len(mesh_data.raw_rest_mesh_data) == 8
    assert "loc2glob_el" in mesh_data.raw_rest_mesh_data[0]
