import numpy as np

from hdg_postprocess.core.solution import preparation as prep_ops
from hdg_postprocess.routines.plasma import (
    calculate_M_cons,
    calculate_Te_cons,
    calculate_Ti_cons,
    calculate_cs_cons,
    calculate_grad_Te_cons,
    calculate_grad_Ti_cons,
    calculate_grad_pe_cons,
    calculate_grad_pi_cons,
    calculate_n_cons,
    calculate_nn_cons,
    calculate_pe_cons,
    calculate_pi_cons,
    calculate_u_cons,
)


_BATCHED_VARIABLE_REQUIREMENTS = {
    "n": {"cons": (b"rho",)},
    "nn": {"cons": (b"rhon",)},
    "te": {"cons": (b"rho", b"nEe")},
    "ti": {"cons": (b"rho", b"Gamma", b"nEi")},
    "u": {"cons": (b"rho", b"Gamma")},
    "cs": {"cons": (b"rho", b"Gamma", b"nEi", b"nEe")},
    "M": {"cons": (b"rho", b"Gamma", b"nEi", b"nEe")},
    "pi": {"cons": (b"rho", b"Gamma", b"nEi")},
    "pe": {"cons": (b"nEe",)},
    "psi": {"fields": ("psi",)},
    "br": {"fields": ("br",)},
    "bz": {"fields": ("bz",)},
    "btor": {"fields": ("btor",)},
    "dpi_dx": {"cons": (b"rho", b"Gamma", b"nEi"), "grad": (b"rho", b"Gamma", b"nEi")},
    "dpi_dy": {"cons": (b"rho", b"Gamma", b"nEi"), "grad": (b"rho", b"Gamma", b"nEi")},
    "dpe_dx": {"grad": (b"nEe",)},
    "dpe_dy": {"grad": (b"nEe",)},
    "dti_dx": {"cons": (b"rho", b"Gamma", b"nEi"), "grad": (b"rho", b"Gamma", b"nEi")},
    "dti_dy": {"cons": (b"rho", b"Gamma", b"nEi"), "grad": (b"rho", b"Gamma", b"nEi")},
    "dte_dx": {"cons": (b"rho", b"nEe"), "grad": (b"rho", b"nEe")},
    "dte_dy": {"cons": (b"rho", b"nEe"), "grad": (b"rho", b"nEe")},
}

_FIELD_INTERPOLATOR_NAMES = {
    "psi": lambda interpolators: interpolators.psi,
    "br": lambda interpolators: interpolators.field[0],
    "bz": lambda interpolators: interpolators.field[1],
    "btor": lambda interpolators: interpolators.field[2],
}


def supported_variables(variable_names):
    return [variable for variable in variable_names if variable in _BATCHED_VARIABLE_REQUIREMENTS]


def sample_variables(solution, r_values, z_values, variable_names):
    variables = supported_variables(variable_names)
    if not variables:
        return {}

    prep_ops.ensure_interpolators(solution)
    x_values = np.asarray(r_values, dtype=np.float64).reshape(-1)
    y_values = np.asarray(z_values, dtype=np.float64).reshape(-1)
    state, gradients, fields = _sample_inputs(solution, x_values, y_values, variables)
    return _compute_variables(solution, state, gradients, fields, x_values, y_values, variables)


def sample_variable_at_point(solution, r, z, variable):
    values = sample_variables(solution, [r], [z], [variable])
    if variable not in values:
        raise KeyError(variable)
    return values[variable]


def _sample_inputs(solution, x_values, y_values, variables):
    cons_names = []
    grad_names = []
    field_names = []
    for variable in variables:
        requirements = _BATCHED_VARIABLE_REQUIREMENTS[variable]
        for cons_name in requirements.get("cons", ()):
            if cons_name not in cons_names:
                cons_names.append(cons_name)
        for grad_name in requirements.get("grad", ()):
            if grad_name not in grad_names:
                grad_names.append(grad_name)
        for field_name in requirements.get("fields", ()):
            if field_name not in field_names:
                field_names.append(field_name)

    state = np.zeros((x_values.size, solution.neq), dtype=np.float64)
    gradients = np.zeros((x_values.size, solution.neq, 2), dtype=np.float64) if grad_names else None

    for cons_name in cons_names:
        idx = solution._cons_idx[cons_name]
        state[:, idx] = solution.interpolators.solution[idx].evaluate_many(x_values, y_values)

    if gradients is not None:
        for grad_name in grad_names:
            idx = solution._cons_idx[grad_name]
            gradients[:, idx, 0] = solution.interpolators.gradient[idx][0].evaluate_many(x_values, y_values)
            gradients[:, idx, 1] = solution.interpolators.gradient[idx][1].evaluate_many(x_values, y_values)

    fields = {}
    for field_name in field_names:
        fields[field_name] = _FIELD_INTERPOLATOR_NAMES[field_name](solution.interpolators).evaluate_many(x_values, y_values)

    return state, gradients, fields


def _compute_variables(solution, state, gradients, fields, x_values, y_values, variables):
    adim = solution.parameters["adimensionalization"]
    physics = solution.parameters["physics"]
    cons_idx = solution._cons_idx
    result = {}
    zero_density_mask = None
    if b"rho" in cons_idx:
        zero_density_mask = state[:, cons_idx[b"rho"]] == 0

    with np.errstate(divide="ignore", invalid="ignore"):
        variable_set = set(variables)
        if "n" in variable_set:
            result["n"] = np.asarray(calculate_n_cons(state, adim["density_scale"], cons_idx), dtype=float)
        if "nn" in variable_set:
            result["nn"] = np.asarray(calculate_nn_cons(state, adim["density_scale"], cons_idx), dtype=float)
        if "te" in variable_set:
            result["te"] = np.asarray(calculate_Te_cons(state, adim["temperature_scale"], physics["Mref"], cons_idx), dtype=float)
        if "ti" in variable_set:
            result["ti"] = np.asarray(calculate_Ti_cons(state, adim["temperature_scale"], physics["Mref"], cons_idx), dtype=float)
        if "u" in variable_set:
            result["u"] = np.asarray(calculate_u_cons(state, adim["speed_scale"], cons_idx), dtype=float)
        if "cs" in variable_set:
            result["cs"] = np.asarray(calculate_cs_cons(state, adim["speed_scale"], cons_idx), dtype=float)
        if "M" in variable_set:
            result["M"] = np.asarray(calculate_M_cons(state, cons_idx), dtype=float)
        if "pi" in variable_set:
            result["pi"] = np.asarray(calculate_pi_cons(state, _pressure_scale(solution), cons_idx), dtype=float)
        if "pe" in variable_set:
            result["pe"] = np.asarray(calculate_pe_cons(state, _pressure_scale(solution), cons_idx), dtype=float)
        if "dpi_dx" in variable_set or "dpi_dy" in variable_set:
            grad_pi = np.asarray(
                calculate_grad_pi_cons(state, gradients, _pressure_scale(solution), adim["length_scale"], cons_idx),
                dtype=float,
            )
            if "dpi_dx" in variable_set:
                result["dpi_dx"] = grad_pi[:, 0]
            if "dpi_dy" in variable_set:
                result["dpi_dy"] = grad_pi[:, 1]
        if "dpe_dx" in variable_set or "dpe_dy" in variable_set:
            grad_pe = np.asarray(calculate_grad_pe_cons(gradients, _pressure_scale(solution), adim["length_scale"], cons_idx), dtype=float)
            if "dpe_dx" in variable_set:
                result["dpe_dx"] = grad_pe[:, 0]
            if "dpe_dy" in variable_set:
                result["dpe_dy"] = grad_pe[:, 1]
        if "dti_dx" in variable_set or "dti_dy" in variable_set:
            grad_ti = np.asarray(
                calculate_grad_Ti_cons(state, gradients, adim["temperature_scale"], physics["Mref"], adim["length_scale"], cons_idx),
                dtype=float,
            )
            if "dti_dx" in variable_set:
                result["dti_dx"] = grad_ti[:, 0]
            if "dti_dy" in variable_set:
                result["dti_dy"] = grad_ti[:, 1]
        if "dte_dx" in variable_set or "dte_dy" in variable_set:
            grad_te = np.asarray(
                calculate_grad_Te_cons(state, gradients, adim["temperature_scale"], physics["Mref"], adim["length_scale"], cons_idx),
                dtype=float,
            )
            if "dte_dx" in variable_set:
                result["dte_dx"] = grad_te[:, 0]
            if "dte_dy" in variable_set:
                result["dte_dy"] = grad_te[:, 1]

    if "psi" in variables:
        result["psi"] = np.asarray(fields["psi"], dtype=float)
    if "br" in variables:
        result["br"] = np.asarray(fields["br"], dtype=float)
    if "bz" in variables:
        result["bz"] = np.asarray(fields["bz"], dtype=float)
    if "btor" in variables:
        result["btor"] = np.asarray(fields["btor"], dtype=float)

    if zero_density_mask is not None and np.any(zero_density_mask):
        for name in ("te", "ti", "u", "cs", "M", "pi", "dpi_dx", "dpi_dy", "dti_dx", "dti_dy", "dte_dx", "dte_dy"):
            if name in result:
                values = np.asarray(result[name], dtype=float).copy()
                values[zero_density_mask] = 0.0
                result[name] = values
    return result


def _pressure_scale(solution):
    return (
        (2 / 3 / solution.parameters["physics"]["Mref"])
        * solution.parameters["adimensionalization"]["density_scale"]
        * solution.parameters["adimensionalization"]["temperature_scale"]
        * solution.parameters["adimensionalization"]["charge_scale"]
    )
