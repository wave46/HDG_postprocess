import numpy as np

from hdg_postprocess.formats import load_from_file
from hdg_postprocess.api import configure_solution_setup
from hdg_postprocess.core.solution import preparation as prep_ops
from hdg_postprocess.core.solution.transport_postprocess import _gauss_grad_psi

from helpers import require_scenario_data, scenario_map


def test_surface_te_methods_return_finite_values(manifest_path):
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

    configure_solution_setup(
        sol,
        reference_element=cfg["reference_element"],
        radiation_model=cfg["radiation_model"],
        atomic_data_dir="demos/data/atomic",
        neutral_diffusion=True,
    )
    sol.parameters["physics"]["R_E"] = cfg["r_e_override"]

    te_node = sol.flux_surface.te(0.8, method="node_band", width=2e-3)
    te_gauss = sol.flux_surface.te(0.8, method="gauss_band", width=2e-3)
    te_shell = sol.flux_surface.te(0.8, method="gauss_shell", width=2e-3)
    delta_te = sol.flux_surface.delta_te(rho_inner=0.8, rho_outer=0.99, method="gauss_shell", width=2e-3)
    minor_radius = sol.flux_surface.minor_radius(0.8, method="gauss_shell", width=2e-3)
    major_radius = sol.flux_surface.major_radius(0.8, method="gauss_shell", width=2e-3)
    epsilon = sol.flux_surface.epsilon(0.8, method="gauss_shell", width=2e-3)
    q_value = sol.flux_surface.q(0.8, method="gauss_shell", width=2e-3)
    x_omp = sol.flux_surface.omp_distance(np.array([0.8, 0.9, 1.0]), n_points=200)
    x_sep = sol.flux_surface.omp_distance(1.0, n_points=200)
    sion = sol.flux_surface.average("sion", 0.8, method="gauss_shell", width=2e-3)
    q_par = sol.flux_surface.average("q_par", 0.8, method="gauss_shell", width=2e-3)
    abs_q_par = sol.flux_surface.average("abs_q_par", 0.8, method="gauss_shell", width=2e-3)
    boundary = sol.analysis.boundary_summary()
    rho_boundary = np.sqrt(np.clip(boundary["psi"], 0.0, None))
    rho_wall = float(np.nanmedian(rho_boundary))
    abs_q_dep = sol.flux_surface.boundary_average("abs_q_dep", rho_wall, width=1e-2)
    collisionality = sol.flux_surface.collisionality(0.8, method="gauss_shell", width=2e-3)
    pinch_factor = sol.flux_surface.pinch_factor(0.8, method="gauss_shell", width=2e-3)
    pinch_velocity = sol.flux_surface.pinch_velocity(0.8, 1.0, rho_edge=0.99, method="gauss_shell", width=2e-3)
    pinch_velocity_geometric = sol.flux_surface.pinch_velocity(
        0.8, 1.0, model="geometric", coefficient=0.3, rho_edge=0.99, method="gauss_shell", width=2e-3
    )

    assert np.isfinite(te_node)
    assert np.isfinite(te_gauss)
    assert np.isfinite(te_shell)
    assert np.isfinite(delta_te)
    assert np.isfinite(minor_radius)
    assert np.isfinite(major_radius)
    assert np.isfinite(epsilon)
    assert np.isfinite(q_value)
    assert x_omp.shape == (3,)
    assert np.isfinite(x_omp).all()
    assert np.isclose(x_sep, 0.0)
    assert np.isfinite(sion)
    assert np.isfinite(q_par)
    assert np.isfinite(abs_q_par)
    assert abs_q_par >= 0.0
    assert np.isfinite(abs_q_dep)
    assert abs_q_dep >= 0.0
    assert np.isfinite(collisionality)
    assert np.isfinite(pinch_factor)
    assert np.isfinite(pinch_velocity)
    assert np.isfinite(pinch_velocity_geometric)
    assert pinch_velocity_geometric >= 0.0


def test_transport_bohm_gyrobohm_returns_finite_profiles(manifest_path):
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

    configure_solution_setup(
        sol,
        reference_element=cfg["reference_element"],
        radiation_model=cfg["radiation_model"],
        atomic_data_dir="demos/data/atomic",
        neutral_diffusion=True,
    )
    sol.parameters["physics"]["R_E"] = cfg["r_e_override"]

    rho = np.linspace(0.3, 0.95, 8)
    bohm = sol.transport.bohm(rho, rho_edge=0.99, method="gauss_shell", width=2e-3)
    gyrobohm = sol.transport.gyrobohm(rho, rho_edge=0.99, method="gauss_shell", width=2e-3)
    profiles = sol.transport.bohm_gyrobohm(rho, rho_edge=0.99, method="gauss_shell", width=2e-3)

    assert np.isfinite(bohm["chi_bohm"]).all()
    assert np.isfinite(gyrobohm["chi_gyrobohm"]).all()
    assert np.isfinite(profiles["chi_i"]).all()
    assert (bohm["chi_bohm"] >= 0.0).all()
    assert (gyrobohm["chi_gyrobohm"] >= 0.0).all()
    assert np.isfinite(profiles["chi_e"]).all()
    assert np.isfinite(profiles["diffusion"]).all()


def test_flux_surface_projection_returns_plot_ready_fields(manifest_path):
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

    configure_solution_setup(
        sol,
        reference_element=cfg["reference_element"],
        radiation_model=cfg["radiation_model"],
        atomic_data_dir="demos/data/atomic",
        neutral_diffusion=True,
    )
    sol.parameters["physics"]["R_E"] = cfg["r_e_override"]

    rho = np.linspace(0.3, 0.95, 8)
    profiles = sol.transport.bohm_gyrobohm(rho, rho_edge=0.99, method="gauss_shell", width=2e-3)

    node_field = sol.flux_surface.project(rho, profiles["chi_i"], target="node")
    gauss_field = sol.flux_surface.project(rho, profiles["chi_i"], target="gauss")

    assert node_field.shape == sol.views.simple.equilibrium.poloidal_flux.shape
    assert gauss_field.shape == sol.views.gauss.equilibrium.poloidal_flux.shape
    assert np.isfinite(node_field).all()
    assert np.isfinite(gauss_field).all()


def test_flux_surface_projection_ignores_nonfinite_profile_points(manifest_path):
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

    configure_solution_setup(
        sol,
        reference_element=cfg["reference_element"],
        radiation_model=cfg["radiation_model"],
        atomic_data_dir="demos/data/atomic",
        neutral_diffusion=True,
    )
    sol.parameters["physics"]["R_E"] = cfg["r_e_override"]

    rho = np.linspace(0.0, 1.0, 8)
    values = np.linspace(1.0, 2.0, 8)
    values[0] = np.nan

    projected = sol.flux_surface.project(rho, values, target="node")

    assert np.isfinite(projected).all()


def test_pointwise_flux_normal_gradients_are_finite(manifest_path):
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

    configure_solution_setup(
        sol,
        reference_element=cfg["reference_element"],
        radiation_model=cfg["radiation_model"],
        atomic_data_dir="demos/data/atomic",
        neutral_diffusion=True,
    )
    sol.parameters["physics"]["R_E"] = cfg["r_e_override"]

    r = float(sol.mesh.global_state.vertices[:, 0].mean())
    z = float(sol.mesh.global_state.vertices[:, 1].mean())

    normal = sol.pointwise.fields.flux_normal(r, z)
    grad_psi_x = sol.pointwise.gradients.psi(r, z, "x")
    grad_psi_y = sol.pointwise.gradients.psi(r, z, "y")
    grad_te_n = sol.pointwise.gradients.te_flux_normal(r, z)
    grad_pe_n = sol.pointwise.gradients.pe_flux_normal(r, z)

    assert np.isfinite(normal).all()
    assert np.isfinite(grad_psi_x)
    assert np.isfinite(grad_psi_y)
    assert np.isfinite(grad_te_n)
    assert np.isfinite(grad_pe_n)


def test_transport_gauss_grad_psi_matches_interpolator(manifest_path):
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

    configure_solution_setup(
        sol,
        reference_element=cfg["reference_element"],
        radiation_model=cfg["radiation_model"],
        atomic_data_dir="demos/data/atomic",
        neutral_diffusion=True,
    )
    sol.parameters["physics"]["R_E"] = cfg["r_e_override"]

    sol.assembly.full()
    sol.assembly.gauss()
    prep_ops.ensure_interpolators(sol)
    grad_psi_gauss = _gauss_grad_psi(sol)
    reference_element = sol.mesh.metadata.reference_element
    connectivity = sol.mesh.global_state.connectivity
    vertices = sol.mesh.global_state.vertices
    coords = np.einsum("ij,kjd->kid", reference_element["N"], vertices[connectivity])

    sample_indices = [(0, 0), (0, -1), (min(5, coords.shape[0] - 1), 0)]
    for elem_idx, gauss_idx in sample_indices:
        r = float(coords[elem_idx, gauss_idx, 0])
        z = float(coords[elem_idx, gauss_idx, 1])
        interpolated = np.asarray(sol.interpolators.psi.gradient(r, z), dtype=float)
        reconstructed = grad_psi_gauss[elem_idx, gauss_idx, :]
        assert np.allclose(reconstructed, interpolated, rtol=1e-6, atol=1e-6)
