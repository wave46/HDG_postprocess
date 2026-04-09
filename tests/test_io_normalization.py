import json

import h5py
import numpy as np

from hdg_postprocess.HDG_solution import HDGsolution
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


def test_solution_transport_1d_facade_exposes_optional_arrays():
    parameters = {
        "Neq": np.array([1]),
        "Ndim": np.array([1]),
        "physics": {
            "physical_variable_names": [b"rho"],
            "conservative_variable_names": [b"rho", b"Gamma", b"nEi", b"nEe", b"rhon"],
            "diff_n": np.array([10.0]),
            "diff_u": np.array([20.0]),
            "diff_e": np.array([30.0]),
            "diff_ee": np.array([40.0]),
            "Mref": np.array([2.0]),
        },
        "adimensionalization": {
            "specific_energy_density_scale": 1.0,
            "time_scale": 1.0,
            "mass_scale": 1.0,
            "length_scale": 2.0,
            "diffusion_scale": 8.0,
            "speed_scale": 4.0,
            "density_scale": 5.0,
            "temperature_scale": 7.0,
            "charge_scale": 3.0,
        },
    }

    sol = HDGsolution(
        raw_solutions=[np.zeros((1, 1))],
        raw_solutions_skeleton=[np.zeros((1, 1))],
        raw_gradients=[np.zeros((1, 1))],
        raw_equilibriums=[{}],
        raw_solution_boundary_infos=[{}],
        parameters=parameters,
        n_partitions=1,
        mesh=object(),
        raw_transport_1d=[{
            "profiles": {
                "rho_grid": np.array([0.0, 0.25, 0.5, 0.75]),
                "shell_weight": np.arange(4.0),
                "U_fs": np.array(
                    [
                        [2.0, 1.0, 10.0, 12.0, 8.0],
                        [4.0, 2.0, 16.0, 18.0, 14.0],
                        [5.0, 1.0, 20.0, 22.0, 17.0],
                        [6.0, 3.0, 26.0, 24.0, 18.0],
                    ]
                ),
                "Q_rad_fs": np.array(
                    [
                        [0.4, 0.1, 0.5, 0.6, 0.7],
                        [0.5, 0.2, 0.6, 0.8, 0.9],
                        [0.2, 0.1, 0.4, 0.7, 0.6],
                        [0.3, 0.2, 0.5, 0.9, 0.8],
                    ]
                ),
            },
            "coefficients": {
                "chi_i_fs": np.linspace(1.0, 2.0, 4),
                "chi_e_fs": np.linspace(2.0, 3.0, 4),
                "d_fs": np.linspace(3.0, 4.0, 4),
                "nu_mom_fs": np.linspace(4.0, 5.0, 4),
                "vpinch_fs": np.linspace(-1.0, -2.0, 4),
            },
            "params": {
                "rho_edge": np.array([0.99]),
                "rho_diffusion_model_max": np.array([0.6]),
                "rho_blend_width": np.array([0.2]),
                "diff_n_min": np.array([2.5]),
                "diff_u_min": np.array([4.5]),
                "diff_e_min": np.array([1.5]),
                "diff_ee_min": np.array([2.0]),
                "rho_pinch_axis_width": np.array([0.2]),
                "rho_pinch_model_max": np.array([0.7]),
                "rho_pinch_edge_width": np.array([0.1]),
            },
        }],
    )

    assert sol.transport_1d.available is True
    assert sol.transport_1d.rho_grid.shape == (4,)
    assert sol.transport_1d.shell_weight.shape == (4,)
    assert sol.transport_1d.U_fs.shape == (4, 5)
    assert sol.transport_1d.Q_rad_fs.shape == (4, 5)
    assert sol.transport_1d.Q_fs is None
    assert sol.transport_1d.chi_i_fs.shape == (4,)
    assert sol.transport_1d.chi_e_fs.shape == (4,)
    assert sol.transport_1d.d_fs.shape == (4,)
    assert sol.transport_1d.nu_mom_fs.shape == (4,)
    assert sol.transport_1d.vpinch_fs.shape == (4,)
    assert sol.transport_1d.params["rho_edge"].shape == (1,)
    assert sol.transport_1d.profiles.rho_grid.shape == (4,)
    assert sol.transport_1d.coefficients.d_fs.shape == (4,)
    assert sol.transport_1d.params.get("rho_edge").shape == (1,)
    assert sol.transport_1d.coefficient_names == ("chi_i_fs", "chi_e_fs", "d_fs", "nu_mom_fs", "vpinch_fs")
    assert sol.transport_1d.derived_names == ("ne_fs", "te_fs", "ti_fs", "pe_fs", "pi_fs", "dte_dr_fs", "dpe_dr_fs")

    raw = sol.transport_1d.raw_profiles(dimensional=False)
    assert np.allclose(raw["d_fs"], np.linspace(3.0, 4.0, 4))

    raw_dim = sol.transport_1d.raw_profiles(dimensional=True)
    assert np.allclose(raw_dim["d_fs"], np.linspace(24.0, 32.0, 4))
    assert np.allclose(raw_dim["vpinch_fs"], np.linspace(-4.0, -8.0, 4))

    effective = sol.transport_1d.effective_profiles(dimensional=False)
    assert np.allclose(effective["d_fs"], np.array([2.5, 3.333333333333333, 6.833333333333333, 10.0]))
    assert np.allclose(effective["vpinch_fs"], np.array([0.0, -1.3333333333333333, -1.6666666666666665, 0.0]))

    effective_dim = sol.transport_1d.effective_profiles(dimensional=True)
    assert np.allclose(effective_dim["d_fs"], np.array([20.0, 26.666666666666664, 54.666666666666664, 80.0]))
    assert np.allclose(effective_dim["vpinch_fs"], np.array([0.0, -5.333333333333333, -6.666666666666666, 0.0]))
    assert np.allclose(sol.transport_1d.profile("d_fs"), effective_dim["d_fs"])
    assert np.allclose(sol.transport_1d.coefficient_profiles(effective=False, dimensional=False)["d_fs"], raw["d_fs"])
    assert np.allclose(sol.transport_1d.coefficient_profiles(effective=True, dimensional=True)["d_fs"], effective_dim["d_fs"])

    derived = sol.transport_1d.derived_profiles(dimensional=False)
    assert np.allclose(derived["ne_fs"], np.array([2.0, 4.0, 5.0, 6.0]))
    assert np.allclose(derived["te_fs"], np.array([2.0, 1.5, 1.4666666666666666, 1.3333333333333333]))
    assert np.allclose(derived["ti_fs"], np.array([1.625, 1.2916666666666667, 1.3266666666666667, 1.4027777777777777]))
    assert np.allclose(derived["pe_fs"], np.array([4.0, 6.0, 7.333333333333333, 8.0]))
    assert np.allclose(derived["pi_fs"], np.array([3.25, 5.166666666666667, 6.633333333333334, 8.416666666666666]))
    assert np.allclose(derived["dpe_dr_fs"], np.array([0.2, 0.26666666666666666, 0.2333333333333333, 0.3]))
    assert np.allclose(derived["dte_dr_fs"], np.array([-0.3, -0.12083333333333332, -0.011999999999999992, -0.016666666666666663]))

    derived_dim = sol.transport_1d.derived_profiles(dimensional=True)
    assert np.allclose(derived_dim["ne_fs"], derived["ne_fs"] * 5.0)
    assert np.allclose(derived_dim["te_fs"], derived["te_fs"] * 7.0)
    assert np.allclose(derived_dim["dte_dr_fs"], derived["dte_dr_fs"] * 3.5)
