import numpy as np


def ensure_sampling_cache(solution):
    """Prepare the solution sampling/interpolator cache for export evaluation."""

    solution.sample.define_interpolators()
    return solution.interpolators


def evaluate_interpolator_on_grid(interpolator, r_grid, z_grid, *, locator=None, outside_value=np.nan):
    """Evaluate one cached point interpolator on a rectangular mesh."""

    values = np.empty_like(r_grid, dtype=float)
    for index in np.ndindex(r_grid.shape):
        r_value = r_grid[index]
        z_value = z_grid[index]
        if locator is not None and int(locator(r_value, z_value)) == -1:
            values[index] = outside_value
            continue
        values[index] = interpolator(r_value, z_value)
    return values


def evaluate_interpolators_on_grid(interpolators, r_grid, z_grid, *, locator=None, outside_value=np.nan):
    """Evaluate several cached point interpolators on a rectangular mesh in one pass."""

    flat_r = r_grid.reshape(-1)
    flat_z = z_grid.reshape(-1)
    valid_mask = np.ones(flat_r.shape, dtype=bool)
    if locator is not None:
        for index, (r_value, z_value) in enumerate(zip(flat_r, flat_z)):
            valid_mask[index] = int(locator(r_value, z_value)) != -1

    values = {}
    for name, interpolator in interpolators.items():
        flat_values = np.full(flat_r.shape, outside_value, dtype=float)
        if np.any(valid_mask):
            flat_values[valid_mask] = interpolator.evaluate_many(flat_r[valid_mask], flat_z[valid_mask])
        values[name] = flat_values.reshape(r_grid.shape)
    return values


def evaluate_variables_on_grid(solution, r_grid, z_grid, variables, *, locator=None, outside_value=np.nan):
    """Evaluate named point-sampled variables on a grid via the flattened line sampler."""

    sampled = solution.sample.line(r_grid.reshape(-1), z_grid.reshape(-1), list(variables))
    reshaped = {name: np.asarray(values).reshape(r_grid.shape) for name, values in sampled.items()}
    if locator is None:
        return reshaped

    outside_mask = np.zeros(r_grid.shape, dtype=bool)
    for index in np.ndindex(r_grid.shape):
        outside_mask[index] = int(locator(r_grid[index], z_grid[index])) == -1
    if not np.any(outside_mask):
        return reshaped
    for name in reshaped:
        values = reshaped[name].astype(float, copy=True)
        values[outside_mask] = outside_value
        reshaped[name] = values
    return reshaped


def equilibrium_interpolators(solution):
    """Return cached equilibrium interpolators prepared by solution.sample.define_interpolators()."""

    ensure_sampling_cache(solution)
    interpolators = solution.interpolators
    return {
        "br": interpolators.field[0],
        "bz": interpolators.field[1],
        "bphi": interpolators.field[2],
        "psi": interpolators.psi,
        "jphi": interpolators.jtor,
    }
