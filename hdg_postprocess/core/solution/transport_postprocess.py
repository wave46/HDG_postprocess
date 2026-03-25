import numpy as np

from hdg_postprocess.core.solution import preparation as prep_ops
from hdg_postprocess.core.solution import surfaces as surface_ops


_ELECTRON_CHARGE = 1.602176634e-19
_ATOMIC_MASS_UNIT = 1.66053906660e-27


def bohm_profile(
    solution,
    rho,
    *,
    rho_inner=0.8,
    rho_edge=0.99,
    method="gauss_shell",
    width=1e-3,
    derivative_mode="flux_normal",
    ion_mass_amu=2.0,
    ion_charge=1.0,
):
    profiles = surface_transport_inputs(
        solution,
        rho,
        rho_inner=rho_inner,
        rho_edge=rho_edge,
        method=method,
        width=width,
        derivative_mode=derivative_mode,
    )
    profiles["chi_bohm"] = bohm_component(
        profiles,
        ion_mass_amu=ion_mass_amu,
        ion_charge=ion_charge,
    )
    return profiles


def gyrobohm_profile(
    solution,
    rho,
    *,
    rho_inner=0.8,
    rho_edge=0.99,
    method="gauss_shell",
    width=1e-3,
    derivative_mode="flux_normal",
    ion_mass_amu=2.0,
    ion_charge=1.0,
):
    profiles = surface_transport_inputs(
        solution,
        rho,
        rho_inner=rho_inner,
        rho_edge=rho_edge,
        method=method,
        width=width,
        derivative_mode=derivative_mode,
    )
    profiles["chi_gyrobohm"] = gyrobohm_component(
        profiles,
        ion_mass_amu=ion_mass_amu,
        ion_charge=ion_charge,
    )
    return profiles


def mixed_bohm_gyrobohm(
    solution,
    rho,
    *,
    rho_inner=0.8,
    rho_edge=0.99,
    method="gauss_shell",
    width=1e-3,
    derivative_mode="flux_normal",
    ion_mass_amu=2.0,
    ion_charge=1.0,
):
    profiles = surface_transport_inputs(
        solution,
        rho,
        rho_inner=rho_inner,
        rho_edge=rho_edge,
        method=method,
        width=width,
        derivative_mode=derivative_mode,
    )
    chi_bohm = bohm_component(profiles, ion_mass_amu=ion_mass_amu, ion_charge=ion_charge)
    chi_gyrobohm = gyrobohm_component(profiles, ion_mass_amu=ion_mass_amu, ion_charge=ion_charge)
    chi_i, chi_e, diffusion = mixed_transport_from_components(chi_bohm, chi_gyrobohm)
    profiles.update(
        {
            "chi_bohm": chi_bohm,
            "chi_gyrobohm": chi_gyrobohm,
            "chi_i": chi_i,
            "chi_e": chi_e,
            "diffusion": diffusion,
        }
    )
    return profiles


def surface_transport_inputs(
    solution,
    rho,
    *,
    rho_inner=0.8,
    rho_edge=0.99,
    method="gauss_shell",
    width=1e-3,
    derivative_mode="flux_normal",
):
    rho_values = np.asarray(rho, dtype=float)
    if rho_values.ndim != 1 or rho_values.size < 2:
        raise ValueError("rho must be a one-dimensional array with at least two points")

    te = np.asarray(surface_ops.te_on_surfaces(solution, rho_values, method=method, width=width), dtype=float)
    density = np.asarray(surface_ops.average_on_surfaces(solution, "n", rho_values, method=method, width=width), dtype=float)
    q_value = np.asarray(surface_ops.q_on_surfaces(solution, rho_values, method=method, width=width), dtype=float)
    btor = np.asarray(surface_ops.average_on_surfaces(solution, "btor", rho_values, method=method, width=width), dtype=float)
    minor_radius = np.asarray(surface_ops.minor_radius_on_surfaces(solution, rho_values, method=method, width=width), dtype=float)
    edge_minor_radius = float(surface_ops.minor_radius_on_surfaces(solution, rho_edge, method=method, width=width))
    delta_te = float(
        surface_ops.delta_te_on_surfaces(
            solution,
            rho_inner=rho_inner,
            rho_outer=rho_edge,
            method=method,
            width=width,
        )
    )

    pressure_like = density * te
    a_over_te_dte, a_over_pe_dpe = radial_gradient_inputs(
        solution,
        rho_values,
        edge_minor_radius=edge_minor_radius,
        method=method,
        width=width,
        derivative_mode=derivative_mode,
        te=te,
        density=density,
        minor_radius=minor_radius,
        pressure_like=pressure_like,
    )

    return {
        "rho": rho_values,
        "te": te,
        "density": density,
        "q": q_value,
        "btor": btor,
        "minor_radius": minor_radius,
        "edge_minor_radius": edge_minor_radius,
        "delta_te": delta_te,
        "pressure_like": pressure_like,
        "a_over_te_dte": a_over_te_dte,
        "a_over_pe_dpe": a_over_pe_dpe,
        "derivative_mode": derivative_mode,
    }


def radial_gradient_inputs(
    solution,
    rho,
    *,
    edge_minor_radius,
    method,
    width,
    derivative_mode,
    te,
    density,
    minor_radius,
    pressure_like,
):
    if derivative_mode == "profile_fd":
        return _profile_gradient_inputs(te, minor_radius, pressure_like, edge_minor_radius)
    if derivative_mode == "flux_normal":
        return _flux_normal_gradient_inputs(solution, rho, edge_minor_radius=edge_minor_radius, method=method, width=width)
    raise ValueError(f"Unsupported derivative mode: {derivative_mode}")


def bohm_component(profiles, *, ion_mass_amu=2.0, ion_charge=1.0):
    cs = _sound_speed(profiles["te"], ion_mass_amu)
    rho_s = _gyro_radius(cs, profiles["btor"], ion_mass_amu, ion_charge)
    return rho_s * cs * profiles["q"]**2 * profiles["a_over_pe_dpe"] * profiles["delta_te"]


def gyrobohm_component(profiles, *, ion_mass_amu=2.0, ion_charge=1.0):
    cs = _sound_speed(profiles["te"], ion_mass_amu)
    rho_s = _gyro_radius(cs, profiles["btor"], ion_mass_amu, ion_charge)
    return rho_s**2 * cs / profiles["edge_minor_radius"] * profiles["a_over_te_dte"]


def mixed_transport_from_components(chi_bohm, chi_gyrobohm):
    chi_i = 1.6e-4 * chi_bohm + 1.75e-2 * chi_gyrobohm
    chi_e = 8.0e-5 * chi_bohm + 3.5e-2 * chi_gyrobohm
    with np.errstate(divide="ignore", invalid="ignore"):
        diffusion = chi_i * chi_e / (chi_i + chi_e)
    return chi_i, chi_e, diffusion


def _profile_gradient_inputs(te, minor_radius, pressure_like, edge_minor_radius):
    dte_dr = np.gradient(te, minor_radius, edge_order=2)
    dpe_dr = np.gradient(pressure_like, minor_radius, edge_order=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_over_te_dte = -edge_minor_radius * dte_dr / te
        a_over_pe_dpe = -edge_minor_radius * dpe_dr / pressure_like
    return np.maximum(a_over_te_dte, 0.0), np.maximum(a_over_pe_dpe, 0.0)


def _flux_normal_gradient_inputs(solution, rho_values, *, edge_minor_radius, method, width):
    if method == "node_band":
        raise ValueError("flux_normal derivatives require gauss-based surface averaging")

    prep_ops.ensure_full_solution(solution)
    prep_ops.ensure_gauss_solution(solution)
    prep_ops.ensure_gauss_volumes(solution)
    if not solution.metadata.flags.gauss_phys_initialized:
        solution.fields.initialize_physical("gauss")

    rho_local = surface_ops.rho_field(solution, target="gauss")
    weights = solution.mesh.geometry.gauss_volumes

    te_idx = solution._phys_idx[b"Te"]
    n_idx = solution._phys_idx[b"rho"]
    te = np.asarray(solution.views.gauss.solution.physical[:, :, te_idx], dtype=float)
    density = np.asarray(solution.views.gauss.solution.physical[:, :, n_idx], dtype=float)
    grad_te = np.asarray(solution.views.gauss.gradient.physical[:, :, te_idx, :], dtype=float)
    grad_n = np.asarray(solution.views.gauss.gradient.physical[:, :, n_idx, :], dtype=float)
    grad_psi = _gauss_grad_psi(solution)

    psi_norm = np.linalg.norm(grad_psi, axis=-1)
    normal = np.zeros_like(grad_psi)
    mask = psi_norm > 0.0
    normal[mask] = grad_psi[mask] / psi_norm[mask, None]

    dte_dr = np.sum(grad_te * normal, axis=-1)
    dne_dr = np.sum(grad_n * normal, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_over_te_dte_local = -edge_minor_radius * dte_dr / te
        a_over_pe_dpe_local = -edge_minor_radius * (dne_dr / density + dte_dr / te)

    a_over_te_dte_local = np.maximum(a_over_te_dte_local, 0.0)
    a_over_pe_dpe_local = np.maximum(a_over_pe_dpe_local, 0.0)

    a_over_te_dte = _average_local_quantity_on_surfaces(a_over_te_dte_local, rho_local, weights, rho_values, method=method, width=width)
    a_over_pe_dpe = _average_local_quantity_on_surfaces(a_over_pe_dpe_local, rho_local, weights, rho_values, method=method, width=width)
    return a_over_te_dte, a_over_pe_dpe


def _gauss_grad_psi(solution):
    reference_element = solution.mesh.metadata.reference_element
    connectivity = solution.mesh.global_state.connectivity
    vertices = solution.mesh.global_state.vertices
    psi = np.asarray(solution.views.glob.equilibrium.poloidal_flux, dtype=float)

    nxi = reference_element["Nxi"]
    neta = reference_element["Neta"]

    x = vertices[connectivity, 0]
    y = vertices[connectivity, 1]

    j11 = np.einsum("ij,kj->ki", nxi, x)
    j12 = np.einsum("ij,kj->ki", nxi, y)
    j21 = np.einsum("ij,kj->ki", neta, x)
    j22 = np.einsum("ij,kj->ki", neta, y)
    det_j = j11 * j22 - j21 * j12

    inv_j11 = j22 / det_j
    inv_j12 = -j21 / det_j
    inv_j21 = -j12 / det_j
    inv_j22 = j11 / det_j

    dpsi_dxi = np.einsum("ij,kj->ki", nxi, psi)
    dpsi_deta = np.einsum("ij,kj->ki", neta, psi)

    dpsi_dx = inv_j11 * dpsi_dxi + inv_j12 * dpsi_deta
    dpsi_dy = inv_j21 * dpsi_dxi + inv_j22 * dpsi_deta
    return np.stack([dpsi_dx, dpsi_dy], axis=-1)


def _average_local_quantity_on_surfaces(values, rho_local, weights, rho_values, *, method, width):
    results = []
    for rho0 in np.asarray(rho_values, dtype=float):
        mask = np.abs(rho_local - rho0) <= width
        if not np.any(mask):
            results.append(np.nan)
            continue
        if method == "gauss_band":
            results.append(float(np.mean(values[mask])))
            continue
        if method == "gauss_shell":
            local_weights = weights[mask]
            weight_sum = float(np.sum(local_weights))
            if weight_sum <= 0.0:
                results.append(np.nan)
            else:
                results.append(float(np.sum(values[mask] * local_weights) / weight_sum))
            continue
        raise ValueError(f"Unsupported derivative averaging method: {method}")
    return np.asarray(results, dtype=float)


def _sound_speed(te_ev, ion_mass_amu):
    ion_mass = ion_mass_amu * _ATOMIC_MASS_UNIT
    return np.sqrt(np.maximum(te_ev, 0.0) * _ELECTRON_CHARGE / ion_mass)


def _gyro_radius(sound_speed, btor, ion_mass_amu, ion_charge):
    ion_mass = ion_mass_amu * _ATOMIC_MASS_UNIT
    omega_ci = ion_charge * _ELECTRON_CHARGE * np.abs(btor) / ion_mass
    with np.errstate(divide="ignore", invalid="ignore"):
        return sound_speed / omega_ci
