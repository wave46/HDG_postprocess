import h5py
import numpy as np
import pytest

from hdg_postprocess.HDG_mesh import HDGmesh
from hdg_postprocess.HDG_solution import HDGsolution
from hdg_postprocess.core.solution.neutral_flux_limiter import (
    check_saved_diagnostic_identities,
    compare_diagnostics_only_runs,
)


def _parameters():
    return {
        "Neq": np.array([5]),
        "Ndim": np.array([2]),
        "switches": {"ohmicsrc": np.array([0])},
        "time": {"Current_time": np.array([0.0])},
        "adimensionalization": {
            "charge_scale": 1.0,
            "density_scale": 1.0,
            "length_scale": 2.0,
            "mass_scale": 1.0,
            "specific_energy_density_scale": 1.0,
            "temperature_scale": 1.0,
            "time_scale": 2.0,
        },
        "physics": {
            "Mref": 1.0,
            "conservative_variable_names": [b"rho", b"Gamma", b"nEi", b"nEe", b"rhon"],
            "physical_variable_names": [b"rho", b"Ti"],
            "neutral_flux_limiter_mode": b"diagnostics_only",
            "neutral_flux_limiter_eps": 0.0,
            "neutral_flux_limiter_fs_flux_min": 0.0,
            "neutral_flux_limiter_fs_fraction": 1.0,
            "neutral_flux_limiter_gamma": 1.0,
        },
        "numerics": {},
    }


def _mesh():
    return HDGmesh(
        raw_vertices=[np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])],
        raw_connectivity=[np.array([[0, 1, 2], [2, 1, 3]])],
        raw_connectivity_boundary=[np.array([[0, 1], [2, 3]])],
        raw_mesh_numbers=[{"Nelems": 2, "Nextfaces": 2, "Nnodes": 4}],
        raw_boundary_flags=[],
        raw_ghost_elements=[],
        raw_ghost_faces=[],
        mesh_parameters={
            "Ndim": 2,
            "nodes_per_element": 3,
            "nodes_per_face": 2,
            "element_type": "triangle",
        },
        n_partitions=1,
    )


def _solution(with_diagnostics=True):
    u = np.zeros((2, 3, 5))
    u[:, :, 0] = 1.0
    u[:, :, 2] = 1.5
    u[:, :, 3] = 1.5
    u[:, :, 4] = np.array([[0.5, 1.0, 2.0], [2.0, 1.5, 0.25]])

    q = np.zeros((2, 3, 5, 2))
    q[:, :, 4, :] = np.array(
        [
            [[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]],
            [[0.0, 1.0], [2.0, 0.0], [1.0, 1.0]],
        ]
    )

    dnn = np.full((2, 3), 2.0)
    gamma_unlim = dnn * np.linalg.norm(q[:, :, 4, :], axis=-1)
    gamma_max = u[:, :, 4]
    activation_ratio = gamma_unlim / gamma_max
    phi = 1.0 / (1.0 + activation_ratio)
    diagnostics = {
        "Dnn": dnn,
        "Gamma_unlim": gamma_unlim,
        "Gamma_max": gamma_max,
        "activation_ratio": activation_ratio,
        "phi": phi,
        "D_eff": phi * dnn,
        "Gamma_lim": phi * gamma_unlim,
    }

    raw_diagnostics = [{key: value.reshape(-1) for key, value in diagnostics.items()}] if with_diagnostics else [{}]
    sol = HDGsolution(
        [u.reshape(-1)],
        [np.zeros((2 * 2 * 5,))],
        [q.reshape(-1)],
        [{"magnetic_field": np.ones((4, 3))}],
        [{}],
        _parameters(),
        1,
        _mesh(),
        raw_neutral_flux_limiter_diagnostics=raw_diagnostics,
    )
    return sol, diagnostics


def test_neutral_flux_limiter_diagnostics_absent_is_backward_compatible():
    sol, _ = _solution(with_diagnostics=False)

    assert sol.neutral_flux_limiter_diagnostics == {}
    with pytest.raises(ValueError, match="does not contain"):
        sol.neutrals.limiter_diagnostic("Dnn")


def test_neutral_flux_limiter_diagnostics_are_reshaped_and_accessible():
    sol, diagnostics = _solution()

    assert sol.neutral_flux_limiter_diagnostics["Dnn"].shape == (2, 3)
    assert np.allclose(sol.neutrals.Dnn, diagnostics["Dnn"])
    assert np.allclose(sol.neutrals.neutral_phi, diagnostics["phi"])
    assert np.allclose(sol.neutrals.neutral_Deff, diagnostics["D_eff"])
    assert np.allclose(sol.neutrals.neutral_gamma_unlim, diagnostics["Gamma_unlim"])
    assert np.allclose(sol.neutrals.neutral_gamma_lim, diagnostics["Gamma_lim"])
    assert np.allclose(sol.neutrals.neutral_gamma_max, diagnostics["Gamma_max"])
    assert np.allclose(sol.neutrals.neutral_activation_ratio, diagnostics["activation_ratio"])


def test_neutral_flux_limiter_unique_node_average_uses_connectivity():
    sol, diagnostics = _solution()

    averaged = sol.neutrals.limiter_diagnostic("Dnn", view="node")

    assert averaged.shape == (4,)
    assert np.allclose(averaged, [2.0, 2.0, 2.0, 2.0])

    averaged_phi = sol.neutrals.limiter_diagnostic("phi", view="node")
    assert np.isclose(averaged_phi[1], 0.5 * (diagnostics["phi"][0, 1] + diagnostics["phi"][1, 1]))
    assert np.isclose(averaged_phi[2], 0.5 * (diagnostics["phi"][0, 2] + diagnostics["phi"][1, 0]))


def test_saved_neutral_flux_limiter_identity_checks_pass():
    sol, _ = _solution()

    report = check_saved_diagnostic_identities(sol, atol=1.0e-14, rtol=1.0e-14, strict=True)

    assert report["D_eff"]["passed"]
    assert report["Gamma_lim"]["passed"]


def test_neutral_flux_limiter_recomputed_phi_matches_saved_for_constant_dnn():
    sol, diagnostics = _solution()
    diffusion_scale = (
        sol.parameters["adimensionalization"]["length_scale"] ** 2
        / sol.parameters["adimensionalization"]["time_scale"]
    )
    neutral_diffusion = {
        "const": True,
        "dnn_soft": False,
        "dnn_max": 2.0 * diffusion_scale,
        "dnn_min": 0.0,
        "ti_soft": False,
        "ti_min": 1.0e-6,
    }

    report = sol.neutrals.verify_limiter_diagnostics(
        atomic_parameters={},
        neutral_diffusion_parameters=neutral_diffusion,
        atol=1.0e-14,
        rtol=1.0e-14,
        strict=True,
    )

    assert report["phi"]["passed"]
    assert np.allclose(sol.neutrals.recompute_limiter_diagnostics({}, neutral_diffusion)["phi"], diagnostics["phi"])


def test_diagnostics_only_solution_comparison_reports_pass_and_fail(tmp_path):
    off = tmp_path / "off.h5"
    diag = tmp_path / "diag.h5"
    for path, offset in [(off, 0.0), (diag, 0.0)]:
        with h5py.File(path, "w") as h5:
            group = h5.create_group("solution")
            group.create_dataset("u", data=np.array([1.0, 2.0]) + offset)
            group.create_dataset("q", data=np.array([3.0, 4.0]) + offset)
            group.create_dataset("u_tilde", data=np.array([5.0, 6.0]) + offset)

    passed = compare_diagnostics_only_runs(off, diag, strict=True)
    assert passed["u"]["passed"]

    with h5py.File(diag, "r+") as h5:
        h5["solution/u"][1] = 2.1

    failed = compare_diagnostics_only_runs(off, diag, atol=1.0e-12, rtol=1.0e-12)
    assert not failed["u"]["passed"]
    with pytest.raises(AssertionError):
        compare_diagnostics_only_runs(off, diag, atol=1.0e-12, rtol=1.0e-12, strict=True)
