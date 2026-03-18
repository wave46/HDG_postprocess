import numpy as np


def ensure_sampling_cache(solution):
    """Prepare the solution sampling/interpolator cache for export evaluation."""

    solution.sample.define_interpolators()
    return solution.interpolators


def evaluate_interpolator_on_grid(interpolator, r_grid, z_grid):
    """Evaluate one cached point interpolator on a rectangular mesh."""

    values = np.empty_like(r_grid, dtype=float)
    for index in np.ndindex(r_grid.shape):
        values[index] = interpolator(r_grid[index], z_grid[index])
    return values


def evaluate_variables_on_grid(solution, r_grid, z_grid, variables):
    """Evaluate named point-sampled variables on a grid via the flattened line sampler."""

    sampled = solution.sample.line(r_grid.reshape(-1), z_grid.reshape(-1), list(variables))
    return {name: np.asarray(values).reshape(r_grid.shape) for name, values in sampled.items()}


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
