import numpy as np
import pytest

from hdg_postprocess.HDG_solution import HDGsolution
from hdg_postprocess.core.solution import transport_1d_postprocess as transport_1d_ops


def _make_transport_solution(raw_transport_1d=None):
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

    return HDGsolution(
        raw_solutions=[np.zeros((1, 1))],
        raw_solutions_skeleton=[np.zeros((1, 1))],
        raw_gradients=[np.zeros((1, 1))],
        raw_equilibriums=[{}],
        raw_solution_boundary_infos=[{}],
        parameters=parameters,
        n_partitions=1,
        mesh=object(),
        raw_transport_1d=raw_transport_1d,
    )


def _transport_payload():
    return [{
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
    }]


def test_transport_1d_facade_profiles_and_derivations():
    sol = _make_transport_solution(_transport_payload())

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


def test_transport_1d_projection_bundle_uses_requested_view(monkeypatch):
    sol = _make_transport_solution(_transport_payload())

    def fake_rho_field(solution, *, view="simple"):
        return np.full((2, 3), {"simple": 1.0, "full": 2.0, "gauss": 3.0}[view])

    def fake_project(solution, name, *, view="simple", effective=True, dimensional=True):
        offset = {"chi_i_fs": 1.0, "chi_e_fs": 2.0, "d_fs": 3.0, "nu_mom_fs": 4.0, "vpinch_fs": 5.0}[name]
        scale = {"simple": 10.0, "full": 20.0, "gauss": 30.0}[view]
        return np.full((2, 3), scale + offset)

    monkeypatch.setattr(transport_1d_ops, "transport_rho_field", fake_rho_field)
    monkeypatch.setattr(transport_1d_ops, "project_transport_profile", fake_project)

    projected = sol.transport_1d.coefficient_profiles(view="gauss", effective=True, dimensional=True)
    assert projected["rho_grid"].shape == (4,)
    assert np.allclose(projected["rho"], np.full((2, 3), 3.0))
    assert np.allclose(projected["d_fs"], np.full((2, 3), 33.0))
    assert np.allclose(projected["vpinch_fs"], np.full((2, 3), 35.0))


def test_transport_1d_unavailable_solution_fails_cleanly():
    sol = _make_transport_solution(None)

    assert sol.transport_1d.available is False
    assert not sol.transport_1d.profiles
    assert sol.transport_1d.rho_grid is None
    assert sol.transport_1d.get("d_fs") is None

    with pytest.raises(ValueError, match="rho_grid"):
        sol.transport_1d.raw_profiles()


def test_transport_1d_rejects_unsupported_view():
    sol = _make_transport_solution(_transport_payload())

    with pytest.raises(ValueError, match="Unsupported transport projection view"):
        sol.transport_1d.coefficient_profiles(view="node")
