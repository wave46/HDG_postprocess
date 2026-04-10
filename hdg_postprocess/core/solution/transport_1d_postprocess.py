import numpy as np

from hdg_postprocess.core.solution import surfaces as surface_ops


_MODEL_TOL = 1.0e-12
_COEFFICIENT_KEYS = ("chi_i_fs", "chi_e_fs", "d_fs", "nu_mom_fs", "vpinch_fs")
_DIFFUSION_BACKGROUNDS = {
    "chi_i_fs": ("diff_e", "diff_e_min"),
    "chi_e_fs": ("diff_ee", "diff_ee_min"),
    "d_fs": ("diff_n", "diff_n_min"),
    "nu_mom_fs": ("diff_u", "diff_u_min"),
}
_DERIVED_KEYS = ("ne_fs", "te_fs", "ti_fs", "pe_fs", "pi_fs", "dte_dr_fs", "dpe_dr_fs")


def coefficient_keys():
    """Return the saved transport coefficient dataset names."""
    return _COEFFICIENT_KEYS


def derived_keys():
    """Return the derived 1D plasma profile names reconstructed from U_fs/Q_rad_fs."""
    return _DERIVED_KEYS


def raw_transport_profiles(solution, *, dimensional=False):
    """Return saved 1D transport coefficients on the stored rho_grid.

    These are the profiles written by the Fortran transport_1d module before
    runtime blending with background diffusion and before pinch windowing.
    """
    rho_grid = _rho_grid(solution)
    profiles = {"rho_grid": rho_grid}
    for key in _COEFFICIENT_KEYS:
        value = solution.transport_1d.get(key)
        if value is None:
            continue
        values = np.asarray(value, dtype=float).reshape(-1)
        if values.size != rho_grid.size:
            raise ValueError(f"{key} has length {values.size}, expected {rho_grid.size}")
        profiles[key] = _dimensionalize(solution, key, values) if dimensional else values.copy()
    return profiles


def effective_transport_profiles(solution, *, dimensional=False):
    """Return the runtime-effective 1D transport coefficients.

    The returned profiles reconstruct the coefficients actually used by the
    solver by applying the saved blending, floor, and pinch-window parameters.
    """
    raw = raw_transport_profiles(solution, dimensional=False)
    rho_grid = raw["rho_grid"]
    params = solution.transport_1d.params
    physics = solution.parameters["physics"]

    effective = {"rho_grid": rho_grid.copy()}
    blend = _edge_cutoff(
        rho_grid,
        _param_scalar(params, "rho_diffusion_model_max", 1.0),
        _param_scalar(params, "rho_blend_width", 0.0),
    )
    pinch_window = _axis_ramp(
        rho_grid,
        _param_scalar(params, "rho_pinch_axis_width", 0.0),
    ) * _edge_cutoff(
        rho_grid,
        _param_scalar(params, "rho_pinch_model_max", 1.0),
        _param_scalar(params, "rho_pinch_edge_width", 0.0),
    )

    for key in _COEFFICIENT_KEYS:
        if key not in raw:
            continue
        values = raw[key].copy()
        values[rho_grid <= _MODEL_TOL] = 0.0
        if key == "vpinch_fs":
            effective_values = pinch_window * values
        else:
            background_key, floor_key = _DIFFUSION_BACKGROUNDS[key]
            background = float(np.asarray(physics[background_key]).reshape(-1)[0])
            floor = _param_scalar(params, floor_key, 0.0)
            floored = np.maximum(values, floor)
            effective_values = background + blend * (floored - background)
        effective[key] = _dimensionalize(solution, key, effective_values) if dimensional else effective_values

    return effective


def transport_profile(solution, name, *, effective=True, dimensional=True):
    """Return a single 1D transport profile by name."""
    profiles = effective_transport_profiles(solution, dimensional=dimensional) if effective else raw_transport_profiles(solution, dimensional=dimensional)
    if name == "rho_grid":
        return profiles["rho_grid"]
    return np.asarray(profiles[name], dtype=float)


def derived_transport_profiles(solution, *, dimensional=True):
    """Reconstruct derived 1D plasma profiles from saved reduced transport data.

    The returned quantities are derived from U_fs and Q_rad_fs rather than read
    directly from HDF5. This keeps the saved transport output compact while
    still exposing the most useful thermodynamic profiles in postprocessing.
    """
    rho_grid = _rho_grid(solution)
    u_fs = np.asarray(solution.transport_1d.U_fs, dtype=float)
    q_rad_fs = np.asarray(solution.transport_1d.Q_rad_fs, dtype=float)
    if u_fs.ndim != 2 or q_rad_fs.ndim != 2:
        raise ValueError("transport_1d U_fs and Q_rad_fs must be two-dimensional arrays")
    if u_fs.shape[0] != rho_grid.size:
        raise ValueError(f"U_fs has shape {u_fs.shape}, expected first dimension {rho_grid.size}")
    if q_rad_fs.shape[0] != rho_grid.size:
        raise ValueError(f"Q_rad_fs has shape {q_rad_fs.shape}, expected first dimension {rho_grid.size}")

    cons_idx = solution.metadata.indices.conservative
    rho_idx = cons_idx[b"rho"]
    gamma_idx = cons_idx[b"Gamma"]
    nei_idx = cons_idx[b"nEi"]
    nee_idx = cons_idx[b"nEe"]

    rho_u = u_fs[:, rho_idx]
    gamma_u = u_fs[:, gamma_idx]
    nei_u = u_fs[:, nei_idx]
    nee_u = u_fs[:, nee_idx]
    q_rho = q_rad_fs[:, rho_idx]
    q_nee = q_rad_fs[:, nee_idx]

    mref = float(np.asarray(solution.parameters["physics"]["Mref"]).reshape(-1)[0])
    pref = 2.0 / (3.0 * mref)
    with np.errstate(divide="ignore", invalid="ignore"):
        ne_fs = rho_u
        te_fs = pref * nee_u / rho_u
        ti_fs = pref * (nei_u / rho_u - 0.5 * gamma_u**2 / rho_u**2)
        pe_fs = pref * nee_u
        pi_fs = pref * (nei_u - 0.5 * gamma_u**2 / rho_u)
        dpe_dr_fs = pref * q_nee
        dte_dr_fs = pref * (q_nee / rho_u - nee_u * q_rho / rho_u**2)

    derived = {
        "rho_grid": rho_grid.copy(),
        "ne_fs": ne_fs,
        "te_fs": te_fs,
        "ti_fs": ti_fs,
        "pe_fs": pe_fs,
        "pi_fs": pi_fs,
        "dte_dr_fs": dte_dr_fs,
        "dpe_dr_fs": dpe_dr_fs,
    }
    for key, values in list(derived.items()):
        if key == "rho_grid":
            continue
        array = np.asarray(values, dtype=float)
        array[~np.isfinite(array)] = np.nan
        derived[key] = _dimensionalize_derived(solution, key, array) if dimensional else array
    return derived


def transport_coefficients(solution, *, view=None, effective=True, dimensional=True):
    """Return transport coefficients either as 1D profiles or projected fields.

    With ``view=None`` this returns 1D profiles on ``rho_grid``.
    With ``view='simple' | 'full' | 'gauss'`` it returns transport fields
    projected onto the corresponding solution view.
    """
    if view is None:
        return effective_transport_profiles(solution, dimensional=dimensional) if effective else raw_transport_profiles(solution, dimensional=dimensional)
    return projected_transport_profiles(
        solution,
        view=view,
        effective=effective,
        dimensional=dimensional,
    )


def project_transport_profile(solution, name, *, view="simple", effective=True, dimensional=True):
    """Project one saved or effective 1D transport profile onto a solution view."""
    profiles = effective_transport_profiles(solution, dimensional=dimensional) if effective else raw_transport_profiles(solution, dimensional=dimensional)
    normalized_view = _normalize_projection_view(view)
    if normalized_view == "simple":
        return surface_ops.project_profile_to_solution(
            solution,
            profiles["rho_grid"],
            profiles[name],
            target="node",
        )
    if normalized_view == "gauss":
        return surface_ops.project_profile_to_solution(
            solution,
            profiles["rho_grid"],
            profiles[name],
            target="gauss",
        )
    local_rho = transport_rho_field(solution, view=normalized_view)
    return _project_profile_values(profiles["rho_grid"], profiles[name], local_rho)


def projected_transport_profiles(solution, *, view="simple", effective=True, dimensional=True):
    """Project all saved transport coefficients onto a solution view."""
    profiles = effective_transport_profiles(solution, dimensional=dimensional) if effective else raw_transport_profiles(solution, dimensional=dimensional)
    normalized_view = _normalize_projection_view(view)
    projected = {
        "rho_grid": profiles["rho_grid"].copy(),
        "rho": transport_rho_field(solution, view=normalized_view),
    }
    for key in _COEFFICIENT_KEYS:
        if key in profiles:
            projected[key] = project_transport_profile(
                solution,
                key,
                view=normalized_view,
                effective=effective,
                dimensional=dimensional,
            )
    return projected


def transport_rho_field(solution, *, view="simple"):
    """Return the local rho_pol_norm field on the requested solution view."""
    normalized_view = _normalize_projection_view(view)
    if normalized_view == "simple":
        return surface_ops.rho_field(solution, target="node")
    if normalized_view == "gauss":
        return surface_ops.rho_field(solution, target="gauss")
    if normalized_view == "full":
        if not solution.metadata.flags.combined_to_full:
            solution.assembly.full()
        return np.sqrt(np.clip(np.asarray(solution.views.glob.equilibrium.poloidal_flux, dtype=float), 0.0, None))
    raise ValueError(f"Unsupported transport projection view: {normalized_view}")


def _rho_grid(solution):
    rho_grid = solution.transport_1d.rho_grid
    if rho_grid is None:
        raise ValueError("The solution file does not contain transport_1d/profiles/rho_grid")
    return np.asarray(rho_grid, dtype=float).reshape(-1)


def _param_scalar(params, name, default):
    value = params.get(name, default)
    if value is default:
        return float(default)
    return float(np.asarray(value).reshape(-1)[0])


def _normalize_projection_view(view):
    normalized = "full" if view == "glob" else view
    if normalized in {"simple", "full", "gauss"}:
        return normalized
    raise ValueError(f"Unsupported transport projection view: {view}")


def _project_profile_values(rho_grid, values, local_rho):
    rho_grid = np.asarray(rho_grid, dtype=float)
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(rho_grid) & np.isfinite(values)
    if np.count_nonzero(finite_mask) < 2:
        raise ValueError("rho and values must contain at least two finite points")
    order = np.argsort(rho_grid[finite_mask])
    rho_sorted = rho_grid[finite_mask][order]
    values_sorted = values[finite_mask][order]
    clipped_rho = np.clip(np.asarray(local_rho, dtype=float), rho_sorted[0], rho_sorted[-1])
    projected = np.interp(clipped_rho.reshape(-1), rho_sorted, values_sorted)
    return projected.reshape(clipped_rho.shape)


def _dimensionalize(solution, name, values):
    values = np.asarray(values, dtype=float)
    adim = solution.parameters["adimensionalization"]
    if name == "vpinch_fs":
        speed_scale = float(adim.get("speed_scale", adim["length_scale"] / adim["time_scale"]))
        return values * speed_scale
    diffusion_scale = float(adim.get("diffusion_scale", adim["length_scale"] ** 2 / adim["time_scale"]))
    return values * diffusion_scale


def _dimensionalize_derived(solution, name, values):
    values = np.asarray(values, dtype=float)
    adim = solution.parameters["adimensionalization"]
    mref = float(np.asarray(solution.parameters["physics"]["Mref"]).reshape(-1)[0])
    pressure_scale = (2.0 / (3.0 * mref)) * float(adim["density_scale"]) * float(adim["temperature_scale"]) * float(adim["charge_scale"])
    if name == "ne_fs":
        return values * float(adim["density_scale"])
    if name in {"te_fs", "ti_fs"}:
        return values * float(adim["temperature_scale"])
    if name in {"pe_fs", "pi_fs"}:
        return values * pressure_scale
    if name in {"dte_dr_fs"}:
        return values * float(adim["temperature_scale"]) / float(adim["length_scale"])
    if name in {"dpe_dr_fs"}:
        return values * pressure_scale / float(adim["length_scale"])
    return values


def _smoothstep01(s):
    clipped = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _axis_ramp(rho, width):
    rho = np.asarray(rho, dtype=float)
    if width <= _MODEL_TOL:
        return np.where(rho > 0.0, 1.0, 0.0)
    ramp = _smoothstep01(rho / width)
    ramp = np.where(rho <= 0.0, 0.0, ramp)
    ramp = np.where(rho >= width, 1.0, ramp)
    return ramp


def _edge_cutoff(rho, rho_max, width):
    rho = np.asarray(rho, dtype=float)
    if width <= _MODEL_TOL:
        return np.where(rho >= rho_max, 0.0, 1.0)
    rho_start = rho_max - width
    cutoff = np.ones_like(rho, dtype=float)
    cutoff = np.where(rho >= rho_max, 0.0, cutoff)
    transition = (rho > rho_start) & (rho < rho_max)
    cutoff[transition] = 1.0 - _smoothstep01((rho[transition] - rho_start) / width)
    return cutoff
