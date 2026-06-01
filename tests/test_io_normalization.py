import json

import h5py
import numpy as np

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


def test_load_solution_data_extracts_optional_transport_1d_group(tmp_path):
    path = tmp_path / "toy_solution.h5"
    with h5py.File(path, "w") as h5:
        sim = h5.create_group("simulation_parameters")
        sim.create_group("switches").create_dataset("ohmicsrc", data=np.array([0]))
        adim = sim.create_group("adimensionalization")
        adim.create_dataset("time_scale", data=np.array([1.0]))
        phys = sim.create_group("physics")
        phys.create_dataset("n0", data=np.array([1.0]))

        sol = h5.create_group("solution")
        sol.create_dataset("u", data=np.zeros((4, 1)))
        sol.create_dataset("u_tilde", data=np.zeros((4, 1)))
        sol.create_dataset("q", data=np.zeros((4, 1)))

        mag = h5.create_group("magnetic")
        mag.create_dataset("magnetic_field", data=np.zeros((2, 1)))

        mesh = h5.create_group("mesh")
        mesh.create_dataset("boundaryFlag", data=np.array([[1]]))
        mesh.create_dataset("extfaces", data=np.array([[1]]))

        tr = h5.create_group("transport_1d")
        profiles = tr.create_group("profiles")
        profiles.create_dataset("rho_grid", data=np.linspace(0.0, 1.0, 5))
        profiles.create_dataset("shell_weight", data=np.arange(5.0))
        profiles.create_dataset("U_fs", data=np.arange(10.0).reshape(5, 2))
        profiles.create_dataset("Q_rad_fs", data=np.arange(10.0).reshape(5, 2))
        coeffs = tr.create_group("coefficients")
        coeffs.create_dataset("chi_i_fs", data=np.linspace(1.0, 2.0, 5))
        coeffs.create_dataset("d_fs", data=np.linspace(3.0, 4.0, 5))
        params = tr.create_group("params")
        params.create_dataset("rho_edge", data=np.array([0.99]))

    solution_data = load_solution_data(str(tmp_path) + "/", "toy_solution", None, None, 1)

    assert solution_data.raw_transport_1d is not None
    assert np.allclose(solution_data.raw_transport_1d[0]["profiles"]["rho_grid"], np.linspace(0.0, 1.0, 5))
    assert solution_data.raw_transport_1d[0]["profiles"]["U_fs"].shape == (5, 2)
    assert solution_data.raw_transport_1d[0]["profiles"]["Q_rad_fs"].shape == (5, 2)
    assert solution_data.raw_transport_1d[0]["coefficients"]["chi_i_fs"].shape == (5,)
    assert solution_data.raw_transport_1d[0]["params"]["rho_edge"].shape == (1,)


def test_load_solution_data_extracts_optional_neutral_limiter_diagnostics(tmp_path):
    path = tmp_path / "toy_solution.h5"
    with h5py.File(path, "w") as h5:
        sim = h5.create_group("simulation_parameters")
        sim.create_group("switches").create_dataset("ohmicsrc", data=np.array([0]))
        sim.create_group("adimensionalization")
        sim.create_group("physics")

        sol = h5.create_group("solution")
        sol.create_dataset("u", data=np.zeros((6, 1)))
        sol.create_dataset("u_tilde", data=np.zeros((4, 1)))
        sol.create_dataset("q", data=np.zeros((12, 1)))

        mag = h5.create_group("magnetic")
        mag.create_dataset("magnetic_field", data=np.zeros((3, 1)))

        mesh = h5.create_group("mesh")
        mesh.create_dataset("boundaryFlag", data=np.array([[1]]))
        mesh.create_dataset("extfaces", data=np.array([[1]]))

        limiter = h5.create_group("neutral_flux_limiter_diagnostics")
        limiter.create_dataset("Dnn", data=np.arange(6.0))
        limiter.create_dataset("phi", data=np.linspace(0.0, 1.0, 6))

    solution_data = load_solution_data(str(tmp_path) + "/", "toy_solution", None, None, 1)

    assert solution_data.raw_neutral_flux_limiter_diagnostics is not None
    assert np.allclose(solution_data.raw_neutral_flux_limiter_diagnostics[0]["Dnn"], np.arange(6.0))
    assert np.allclose(solution_data.raw_neutral_flux_limiter_diagnostics[0]["phi"], np.linspace(0.0, 1.0, 6))


def test_load_solution_data_extracts_optional_neutral_wall_source_diagnostics(tmp_path):
    path = tmp_path / "toy_solution.h5"
    with h5py.File(path, "w") as h5:
        sim = h5.create_group("simulation_parameters")
        sim.create_group("switches").create_dataset("ohmicsrc", data=np.array([0]))
        sim.create_group("adimensionalization")
        sim.create_group("physics")

        sol = h5.create_group("solution")
        sol.create_dataset("u", data=np.zeros((6, 1)))
        sol.create_dataset("u_tilde", data=np.zeros((4, 1)))
        sol.create_dataset("q", data=np.zeros((12, 1)))

        mag = h5.create_group("magnetic")
        mag.create_dataset("magnetic_field", data=np.zeros((3, 1)))

        mesh = h5.create_group("mesh")
        mesh.create_dataset("boundaryFlag", data=np.array([[1]]))
        mesh.create_dataset("extfaces", data=np.array([[1]]))

        wall = h5.create_group("neutral_wall_sources_diagnostics")
        wall.create_dataset("element_puff_total", data=np.array([3.0]))
        wall.create_dataset("element_pump_total", data=np.array([1.0]))
        wall.create_dataset("element_net_total", data=np.array([2.0]))
        wall.create_dataset("net_flux_density", data=np.arange(6.0))

    solution_data = load_solution_data(str(tmp_path) + "/", "toy_solution", None, None, 1)

    diagnostics = solution_data.raw_neutral_wall_source_diagnostics[0]
    assert diagnostics["element_puff_total"] == 3.0
    assert diagnostics["element_pump_total"] == 1.0
    assert diagnostics["element_net_total"] == 2.0
    assert np.allclose(diagnostics["net_flux_density"], np.arange(6.0))


def test_load_solution_data_extracts_new_diagnostics_wall_sources(tmp_path):
    path = tmp_path / "toy_solution.h5"
    with h5py.File(path, "w") as h5:
        sim = h5.create_group("simulation_parameters")
        sim.create_group("switches").create_dataset("ohmicsrc", data=np.array([0]))
        sim.create_group("adimensionalization")
        sim.create_group("physics")

        sol = h5.create_group("solution")
        sol.create_dataset("u", data=np.zeros((6, 1)))
        sol.create_dataset("u_tilde", data=np.zeros((4, 1)))
        sol.create_dataset("q", data=np.zeros((12, 1)))

        mag = h5.create_group("magnetic")
        mag.create_dataset("magnetic_field", data=np.zeros((3, 1)))

        mesh = h5.create_group("mesh")
        mesh.create_dataset("boundaryFlag", data=np.array([[1]]))
        mesh.create_dataset("extfaces", data=np.array([[1]]))

        wall = h5.create_group("diagnostics/balance/neutrals/particles/nodal_wall_sources")
        wall.create_dataset("element_puff_total", data=np.array([3.0]))
        wall.create_dataset("element_pump_total", data=np.array([1.0]))
        wall.create_dataset("element_net_total", data=np.array([2.0]))
        wall.create_dataset("net_flux_density", data=np.arange(6.0))

    solution_data = load_solution_data(str(tmp_path) + "/", "toy_solution", None, None, 1)

    diagnostics = solution_data.raw_neutral_wall_source_diagnostics[0]
    assert diagnostics["element_puff_total"] == 3.0
    assert diagnostics["element_pump_total"] == 1.0
    assert diagnostics["element_net_total"] == 2.0
    assert np.allclose(diagnostics["net_flux_density"], np.arange(6.0))
