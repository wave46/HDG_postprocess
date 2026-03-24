import numpy as np


_PHYSICAL_FIELDS = {
    "n": b"rho",
    "te": b"Te",
    "ti": b"Ti",
    "u": b"u",
    "mach": b"M",
    "nn": b"rhon",
    "k": b"k",
}



_GEOMETRY_FIELDS = {"minor_radius", "major_radius", "epsilon", "q", "psi", "btor"}


def average_on_surfaces(solution, field, rho, *, method="gauss_shell", width=1e-3):
    rho_values = np.atleast_1d(np.asarray(rho, dtype=float))
    results = np.array([_surface_average_scalar(solution, field, one_rho, method=method, width=width) for one_rho in rho_values])
    if np.asarray(rho).ndim == 0:
        return float(results[0])
    return results


def te_on_surfaces(solution, rho, *, method="gauss_shell", width=1e-3):
    return average_on_surfaces(solution, "te", rho, method=method, width=width)


def delta_te_on_surfaces(solution, *, rho_inner=0.8, rho_outer=1.0, method="gauss_shell", width=1e-3):
    te_inner = te_on_surfaces(solution, rho_inner, method=method, width=width)
    te_outer = te_on_surfaces(solution, rho_outer, method=method, width=width)
    if not np.isfinite(te_inner) or not np.isfinite(te_outer) or np.isclose(te_outer, 0.0):
        return np.nan
    return float((te_inner - te_outer) / te_outer)


def minor_radius_on_surfaces(solution, rho, *, method="gauss_shell", width=1e-3):
    return average_on_surfaces(solution, "minor_radius", rho, method=method, width=width)


def major_radius_on_surfaces(solution, rho, *, method="gauss_shell", width=1e-3):
    return average_on_surfaces(solution, "major_radius", rho, method=method, width=width)


def epsilon_on_surfaces(solution, rho, *, method="gauss_shell", width=1e-3):
    return average_on_surfaces(solution, "epsilon", rho, method=method, width=width)


def q_on_surfaces(solution, rho, *, method="gauss_shell", width=1e-3):
    return average_on_surfaces(solution, "q", rho, method=method, width=width)


def collisionality_on_surfaces(
    solution,
    rho,
    *,
    z_effective=1.0,
    coulomb_logarithm=None,
    method="gauss_shell",
    width=1e-3,
):
    rho_values = np.atleast_1d(np.asarray(rho, dtype=float))
    q_values = np.asarray(q_on_surfaces(solution, rho_values, method=method, width=width), dtype=float)
    r_values = np.asarray(major_radius_on_surfaces(solution, rho_values, method=method, width=width), dtype=float)
    epsilon_values = np.asarray(epsilon_on_surfaces(solution, rho_values, method=method, width=width), dtype=float)
    ne_values = np.asarray(average_on_surfaces(solution, "n", rho_values, method=method, width=width), dtype=float)
    te_values = np.asarray(te_on_surfaces(solution, rho_values, method=method, width=width), dtype=float)

    if coulomb_logarithm is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            coulomb_log = 31.3 - np.log(np.sqrt(ne_values) / te_values)
    else:
        coulomb_log = np.asarray(coulomb_logarithm, dtype=float)
        if coulomb_log.ndim == 0:
            coulomb_log = np.full_like(rho_values, float(coulomb_log))

    with np.errstate(divide="ignore", invalid="ignore"):
        result = (
            6.921e-18
            * q_values
            * r_values
            * ne_values
            * z_effective
            * coulomb_log
            / (te_values**2 * epsilon_values**1.5)
        )

    result = np.asarray(result, dtype=float)
    if np.asarray(rho).ndim == 0:
        return float(result[0])
    return result


def pinch_factor_on_surfaces(
    solution,
    rho,
    *,
    z_effective=1.0,
    coulomb_logarithm=None,
    threshold=0.04,
    method="gauss_shell",
    width=1e-3,
):
    collisionality = np.asarray(
        collisionality_on_surfaces(
            solution,
            rho,
            z_effective=z_effective,
            coulomb_logarithm=coulomb_logarithm,
            method=method,
            width=width,
        ),
        dtype=float,
    )
    factor = np.minimum(1.0, np.exp(1.0 - collisionality / threshold))
    if np.asarray(rho).ndim == 0:
        return float(np.asarray(factor, dtype=float)[0] if np.asarray(factor).ndim > 0 else factor)
    return np.asarray(factor, dtype=float)


def geometric_pinch_velocity_on_surfaces(
    solution,
    rho,
    diffusivity,
    *,
    rho_edge=0.99,
    coefficient=0.5,
    method="gauss_shell",
    width=1e-3,
):
    rho_values = np.atleast_1d(np.asarray(rho, dtype=float))
    diffusivity_values = np.asarray(diffusivity, dtype=float)
    if diffusivity_values.ndim == 0:
        diffusivity_values = np.full_like(rho_values, float(diffusivity_values))
    elif diffusivity_values.shape != rho_values.shape:
        raise ValueError("diffusivity must be scalar or have the same shape as rho")

    minor_radius = np.asarray(minor_radius_on_surfaces(solution, rho_values, method=method, width=width), dtype=float)
    edge_minor_radius = float(minor_radius_on_surfaces(solution, rho_edge, method=method, width=width))

    if np.isclose(edge_minor_radius, 0.0):
        result = np.full_like(rho_values, np.nan, dtype=float)
    else:
        result = coefficient * diffusivity_values * minor_radius / (edge_minor_radius**2)

    if np.asarray(rho).ndim == 0:
        return float(result[0])
    return result


def pinch_velocity_on_surfaces(
    solution,
    rho,
    diffusivity,
    *,
    model="militello",
    rho_edge=0.99,
    coefficient=0.5,
    z_effective=1.0,
    coulomb_logarithm=None,
    threshold=0.04,
    method="gauss_shell",
    width=1e-3,
):
    if model == "geometric":
        return geometric_pinch_velocity_on_surfaces(
            solution,
            rho,
            diffusivity,
            rho_edge=rho_edge,
            coefficient=coefficient,
            method=method,
            width=width,
        )
    if model != "militello":
        raise ValueError(f"Unsupported pinch model: {model}")

    rho_values = np.atleast_1d(np.asarray(rho, dtype=float))
    factor = np.asarray(
        pinch_factor_on_surfaces(
            solution,
            rho_values,
            z_effective=z_effective,
            coulomb_logarithm=coulomb_logarithm,
            threshold=threshold,
            method=method,
            width=width,
        ),
        dtype=float,
    )
    result = factor * np.asarray(
        geometric_pinch_velocity_on_surfaces(
            solution,
            rho_values,
            diffusivity,
            rho_edge=rho_edge,
            coefficient=coefficient,
            method=method,
            width=width,
        ),
        dtype=float,
    )

    if np.asarray(rho).ndim == 0:
        return float(result[0])
    return result


def rho_field(solution, *, target="node"):
    if target == "node":
        return np.sqrt(np.clip(np.asarray(_node_psi(solution), dtype=float), 0.0, None))
    if target == "gauss":
        return np.sqrt(np.clip(np.asarray(_gauss_psi(solution), dtype=float), 0.0, None))
    raise ValueError(f"Unsupported projection target: {target}")


def project_profile_to_solution(solution, rho, values, *, target="node"):
    rho_grid = np.asarray(rho, dtype=float)
    profile_values = np.asarray(values, dtype=float)
    if rho_grid.ndim != 1 or profile_values.ndim != 1:
        raise ValueError("rho and values must be one-dimensional arrays")
    if rho_grid.size != profile_values.size:
        raise ValueError("rho and values must have the same length")
    if rho_grid.size < 2:
        raise ValueError("rho and values must contain at least two points")

    finite_mask = np.isfinite(rho_grid) & np.isfinite(profile_values)
    if np.count_nonzero(finite_mask) < 2:
        raise ValueError("rho and values must contain at least two finite points")

    order = np.argsort(rho_grid[finite_mask])
    rho_sorted = rho_grid[finite_mask][order]
    values_sorted = profile_values[finite_mask][order]
    local_rho = rho_field(solution, target=target)
    clipped_rho = np.clip(local_rho, rho_sorted[0], rho_sorted[-1])
    projected = np.interp(clipped_rho.reshape(-1), rho_sorted, values_sorted)
    return projected.reshape(local_rho.shape)


def _surface_average_scalar(solution, field, rho0, *, method, width):
    if method == "node_band":
        values, rho_values = _node_scalar_and_rho(solution, field)
        return _arithmetic_band_average(values, rho_values, rho0, width)
    if method == "gauss_band":
        values, rho_values = _gauss_scalar_and_rho(solution, field)
        return _arithmetic_band_average(values, rho_values, rho0, width)
    if method == "gauss_shell":
        values, rho_values = _gauss_scalar_and_rho(solution, field)
        weights = _gauss_weights(solution)
        return _weighted_band_average(values, rho_values, weights, rho0, width)
    raise ValueError(f"Unsupported surface averaging method: {method}")


def _node_scalar_and_rho(solution, field):
    scalar = _node_field(solution, field)
    psi = _node_psi(solution)
    return np.asarray(scalar, dtype=float), np.sqrt(np.clip(np.asarray(psi, dtype=float), 0.0, None))


def _gauss_scalar_and_rho(solution, field):
    scalar = _gauss_field(solution, field)
    psi = _gauss_psi(solution)
    return np.asarray(scalar, dtype=float), np.sqrt(np.clip(np.asarray(psi, dtype=float), 0.0, None))


def _node_field(solution, field):
    if field == "psi":
        return _node_psi(solution)
    if field == "major_radius":
        return _node_major_radius(solution)
    if field == "minor_radius":
        return _node_minor_radius(solution)
    if field == "epsilon":
        major_radius = _node_major_radius(solution)
        minor_radius = _node_minor_radius(solution)
        with np.errstate(divide="ignore", invalid="ignore"):
            return minor_radius / major_radius
    if field == "q":
        return _node_q(solution)
    if field == "btor":
        return _node_btor(solution)

    physical_name = _PHYSICAL_FIELDS.get(field)
    if physical_name is None:
        raise KeyError(f"Unsupported flux-surface field: {field}")
    if not solution.metadata.flags.simple_phys_initialized:
        solution.fields.initialize_physical("simple")
    index = solution._phys_idx[physical_name]
    return solution.views.simple.solution.physical[:, index]


def _gauss_field(solution, field):
    if field == "psi":
        return _gauss_psi(solution)
    if field == "major_radius":
        return _gauss_major_radius(solution)
    if field == "minor_radius":
        return _gauss_minor_radius(solution)
    if field == "epsilon":
        major_radius = _gauss_major_radius(solution)
        minor_radius = _gauss_minor_radius(solution)
        with np.errstate(divide="ignore", invalid="ignore"):
            return minor_radius / major_radius
    if field == "q":
        return _gauss_q(solution)
    if field == "btor":
        return _gauss_btor(solution)

    physical_name = _PHYSICAL_FIELDS.get(field)
    if physical_name is None:
        raise KeyError(f"Unsupported flux-surface field: {field}")
    if not solution.metadata.flags.gauss_phys_initialized:
        solution.fields.initialize_physical("gauss")
    index = solution._phys_idx[physical_name]
    return solution.views.gauss.solution.physical[:, :, index]


def _node_psi(solution):
    if not solution.metadata.flags.combined_simple_solution:
        solution.assembly.simple()
    return solution.views.simple.equilibrium.poloidal_flux


def _gauss_psi(solution):
    if not solution.metadata.flags.combined_gauss:
        solution.assembly.gauss()
    return solution.views.gauss.equilibrium.poloidal_flux


def _node_major_radius(solution):
    return solution.mesh.global_state.vertices[:, 0]


def _gauss_major_radius(solution):
    solution.mesh.geometry.gauss_volumes
    return solution.mesh.derived_geometry.vertices_gauss[:, :, 0]


def _node_minor_radius(solution):
    return solution.equilibrium.define_minor_radii(view="simple")


def _gauss_minor_radius(solution):
    full_values = solution.equilibrium.define_minor_radii(view="glob")
    return _interpolate_full_to_gauss(solution, full_values)


def _node_q(solution):
    return solution.equilibrium.define_qcyl(view="simple")


def _node_btor(solution):
    if not solution.metadata.flags.combined_simple_solution:
        solution.assembly.simple()
    return solution.views.simple.equilibrium.magnetic_field[:, 2]


def _gauss_btor(solution):
    if not solution.metadata.flags.combined_gauss:
        solution.assembly.gauss()
    return solution.views.gauss.equilibrium.magnetic_field[:, :, 2]


def _gauss_q(solution):
    full_values = solution.equilibrium.define_qcyl(view="glob")
    return _interpolate_full_to_gauss(solution, full_values)


def _interpolate_full_to_gauss(solution, full_values):
    if not solution.metadata.flags.combined_gauss:
        solution.assembly.gauss()
    return np.einsum("ij,kj->ki", solution.mesh.metadata.reference_element["N"], np.asarray(full_values, dtype=float))


def _gauss_weights(solution):
    return solution.mesh.geometry.gauss_volumes


def _arithmetic_band_average(values, rho_values, rho0, width):
    mask = np.abs(rho_values - rho0) <= width
    if not np.any(mask):
        return np.nan
    return float(np.mean(values[mask]))


def _weighted_band_average(values, rho_values, weights, rho0, width):
    mask = np.abs(rho_values - rho0) <= width
    if not np.any(mask):
        return np.nan
    selected_weights = weights[mask]
    weight_sum = float(np.sum(selected_weights))
    if weight_sum <= 0.0:
        return np.nan
    return float(np.sum(values[mask] * selected_weights) / weight_sum)
