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
        if not solution.metadata.flags.combined_simple_solution:
            solution.assembly.simple()
        return solution.views.simple.equilibrium.poloidal_flux

    physical_name = _PHYSICAL_FIELDS.get(field)
    if physical_name is None:
        raise KeyError(f"Unsupported surface field: {field}")
    if not solution.metadata.flags.simple_phys_initialized:
        solution.fields.initialize_physical("simple")
    index = solution._phys_idx[physical_name]
    return solution.views.simple.solution.physical[:, index]


def _gauss_field(solution, field):
    if field == "psi":
        if not solution.metadata.flags.combined_gauss:
            solution.assembly.gauss()
        return solution.views.gauss.equilibrium.poloidal_flux

    physical_name = _PHYSICAL_FIELDS.get(field)
    if physical_name is None:
        raise KeyError(f"Unsupported surface field: {field}")
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
