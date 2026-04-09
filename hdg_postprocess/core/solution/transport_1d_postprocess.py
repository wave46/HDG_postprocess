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


def coefficient_keys():
    return _COEFFICIENT_KEYS


def raw_transport_profiles(solution, *, dimensional=False):
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
    profiles = effective_transport_profiles(solution, dimensional=dimensional) if effective else raw_transport_profiles(solution, dimensional=dimensional)
    if name == "rho_grid":
        return profiles["rho_grid"]
    return np.asarray(profiles[name], dtype=float)


def project_transport_profile(solution, name, *, target="node", effective=True, dimensional=True):
    profiles = effective_transport_profiles(solution, dimensional=dimensional) if effective else raw_transport_profiles(solution, dimensional=dimensional)
    return surface_ops.project_profile_to_solution(
        solution,
        profiles["rho_grid"],
        profiles[name],
        target=target,
    )


def transport_rho_field(solution, *, target="node"):
    return surface_ops.rho_field(solution, target=target)


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


def _dimensionalize(solution, name, values):
    values = np.asarray(values, dtype=float)
    adim = solution.parameters["adimensionalization"]
    if name == "vpinch_fs":
        speed_scale = float(adim.get("speed_scale", adim["length_scale"] / adim["time_scale"]))
        return values * speed_scale
    diffusion_scale = float(adim.get("diffusion_scale", adim["length_scale"] ** 2 / adim["time_scale"]))
    return values * diffusion_scale


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
