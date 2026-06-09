import numpy as np


def exponential_decay_fit(x, y, *, mask=None, x0=None):
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same shape")

    valid = np.isfinite(x_values) & np.isfinite(y_values) & (y_values > 0.0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two finite positive profile points are required")

    x_fit = x_values[valid]
    y_fit = y_values[valid]
    if x0 is None:
        x0 = float(np.nanmin(x_fit))

    xr = x_fit - float(x0)
    slope, intercept = np.polyfit(xr, np.log(y_fit), 1)
    decay_length = -1.0 / slope
    fitted = np.exp(intercept + slope * xr)
    return {
        "lambda": decay_length,
        "amplitude": float(np.exp(intercept)),
        "slope": float(slope),
        "intercept": float(intercept),
        "x0": float(x0),
        "x": x_fit,
        "xr": xr,
        "y": y_fit,
        "fit": fitted,
    }


def local_decay_length(x, y):
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -1.0 / np.gradient(np.log(y_values), x_values)
