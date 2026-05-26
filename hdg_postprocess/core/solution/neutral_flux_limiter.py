from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np

from hdg_postprocess.api.setup import make_neutral_diffusion_parameters
from hdg_postprocess.routines.neutrals import calculate_dnn_with_nn_collision_cons


DIAGNOSTIC_GROUP = "neutral_flux_limiter_diagnostics"
DIAGNOSTIC_DATASETS = (
    "Dnn",
    "phi",
    "D_eff",
    "Gamma_unlim",
    "Gamma_max",
    "activation_ratio",
    "Gamma_lim",
)
ACCESSOR_NAMES = {
    "Dnn": "Dnn",
    "phi": "neutral_phi",
    "D_eff": "neutral_Deff",
    "Gamma_unlim": "neutral_gamma_unlim",
    "Gamma_lim": "neutral_gamma_lim",
    "Gamma_max": "neutral_gamma_max",
    "activation_ratio": "neutral_activation_ratio",
}
SOLUTION_DATASETS = ("u", "q", "u_tilde")
SUPPORTED_RECOMPUTE_MODES = ("diagnostics_only", "lagged_flux_limiter")


def diagnostic_field(solution, name, view="element"):
    """Return one saved limiter diagnostic as element-node or averaged node data."""
    dataset = _dataset_name(name)
    diagnostics = solution.neutral_flux_limiter_diagnostics
    if not diagnostics:
        raise ValueError("This solution does not contain /neutral_flux_limiter_diagnostics.")
    if dataset not in diagnostics:
        raise KeyError(f"Neutral flux limiter diagnostic '{dataset}' is not available.")
    values = diagnostics[dataset]
    if view in {"element", "full", "element_node"}:
        return values
    if view in {"node", "simple", "unique"}:
        return average_element_nodes_to_unique_nodes(solution, values)
    raise ValueError(f"Unsupported neutral limiter diagnostic view: {view}")


def average_element_nodes_to_unique_nodes(solution, values):
    """Average element-node values onto unique mesh nodes using the full connectivity."""
    if not solution.mesh.metadata.flags.combined_to_full:
        solution.mesh.assembly.full()
    values = np.asarray(values)
    connectivity = solution.mesh.global_state.connectivity
    if values.shape[:2] != connectivity.shape:
        raise ValueError(
            f"Element-node values have shape {values.shape[:2]}, expected mesh connectivity shape {connectivity.shape}."
        )

    trailing_shape = values.shape[2:]
    result = np.zeros((solution.mesh.global_state.n_vertices, *trailing_shape), dtype=values.dtype)
    counts = np.zeros(solution.mesh.global_state.n_vertices, dtype=float)
    np.add.at(result, connectivity.reshape(-1), values.reshape(connectivity.size, *trailing_shape))
    np.add.at(counts, connectivity.reshape(-1), 1.0)
    valid = counts > 0
    result[valid] = result[valid] / counts[valid].reshape((-1,) + (1,) * len(trailing_shape))
    return result


def summarize_diagnostics(solution, activation_tol=1.0e-12):
    """Return min/max and activation fractions for saved limiter diagnostics."""
    diagnostics = solution.neutral_flux_limiter_diagnostics
    if not diagnostics:
        raise ValueError("This solution does not contain /neutral_flux_limiter_diagnostics.")
    summary = {}
    for key, values in diagnostics.items():
        summary[key] = {
            "shape": values.shape,
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)),
        }
    if "phi" in diagnostics:
        summary["fraction_phi_active"] = float(np.mean(diagnostics["phi"] < 1.0 - activation_tol))
    if "activation_ratio" in diagnostics:
        summary["fraction_activation_ratio_gt_one"] = float(np.mean(diagnostics["activation_ratio"] > 1.0))
    return summary


def compare_diagnostics_only_runs(off_path, diagnostics_path, atol=1.0e-12, rtol=1.0e-10, strict=False):
    """Compare solution arrays from neutral-limiter off and diagnostics-only HDF5 runs."""
    report = {}
    with h5py.File(off_path, "r") as off_file, h5py.File(diagnostics_path, "r") as diagnostics_file:
        for dataset in SOLUTION_DATASETS:
            path = f"solution/{dataset}"
            if path not in off_file:
                raise KeyError(f"Missing dataset '/{path}' in off run: {off_path}")
            if path not in diagnostics_file:
                raise KeyError(f"Missing dataset '/{path}' in diagnostics-only run: {diagnostics_path}")
            report[dataset] = _compare_arrays(off_file[path][()], diagnostics_file[path][()], atol, rtol)

    if strict:
        failed = [name for name, item in report.items() if not item["passed"]]
        if failed:
            details = ", ".join(f"{name}: max_abs={report[name]['max_abs_diff']:.3e}" for name in failed)
            raise AssertionError(f"Diagnostics-only solution comparison failed for {details}")
    return report


def recompute_neutral_flux_limiter_diagnostics(
    solution,
    atomic_parameters=None,
    neutral_diffusion_parameters=None,
    flux_convention="diffusion_only",
):
    """Recompute neutral limiter diagnostics independently from saved diagnostic targets."""
    if flux_convention != "diffusion_only":
        raise ValueError(
            f"Unsupported neutral limiter flux convention '{flux_convention}'. "
            "Only 'diffusion_only' is implemented for neutral limiter verification."
        )
    _require_supported_limiter_mode(solution)
    _require_conservative_variables(solution, (b"rho", b"Gamma", b"nEi", b"rhon"))

    if not solution.metadata.flags.combined_to_full:
        solution.assembly.full()

    u = solution.views.glob.solution.conservative
    q = solution.views.glob.gradient.conservative
    physics = solution.parameters["physics"]
    adim = solution.parameters["adimensionalization"]
    neutral_diffusion_parameters = _prepare_neutral_diffusion_parameters(
        neutral_diffusion_parameters,
        adim,
    )
    atomic_parameters = _resolve_atomic_parameters(solution, atomic_parameters)

    dnn_dimensional = calculate_dnn_with_nn_collision_cons(
        u,
        neutral_diffusion_parameters,
        atomic_parameters,
        _scalar(adim, "charge_scale"),
        _scalar(adim, "mass_scale"),
        _scalar(adim, "temperature_scale"),
        _scalar(adim, "density_scale"),
        _scalar(physics, "Mref"),
        _scalar(adim, "length_scale"),
        _scalar(adim, "time_scale"),
    )
    diffusion_scale = _scalar(adim, "length_scale") ** 2 / _scalar(adim, "time_scale")
    dnn = dnn_dimensional / diffusion_scale

    inn = solution.metadata.indices.conservative[b"rhon"]
    gamma_unlim = dnn * np.linalg.norm(q[:, :, inn, :], axis=-1)
    gamma_max = _calculate_gamma_max(solution, u, neutral_diffusion_parameters)
    eps_gamma = _scalar(physics, "neutral_flux_limiter_eps", 0.0)
    gamma_norm = np.sqrt(gamma_unlim**2 + eps_gamma**2)
    activation_ratio = np.divide(
        gamma_norm,
        gamma_max,
        out=np.full_like(gamma_norm, np.inf),
        where=np.abs(gamma_max) > 0.0,
    )
    limiter_gamma = _scalar(physics, "neutral_flux_limiter_gamma")
    phi = (1.0 + activation_ratio**limiter_gamma) ** (-1.0 / limiter_gamma)
    d_eff = phi * dnn
    gamma_lim = phi * gamma_unlim
    return {
        "Dnn": dnn,
        "Gamma_unlim": gamma_unlim,
        "Gamma_max": gamma_max,
        "activation_ratio": activation_ratio,
        "phi": phi,
        "D_eff": d_eff,
        "Gamma_lim": gamma_lim,
    }


def compare_neutral_flux_limiter_diagnostics(
    solution,
    atomic_parameters=None,
    neutral_diffusion_parameters=None,
    atol=1.0e-10,
    rtol=1.0e-8,
    strict=False,
    fields=None,
    flux_convention="diffusion_only",
):
    """Compare independently recomputed limiter diagnostics against saved targets."""
    saved = solution.neutral_flux_limiter_diagnostics
    if not saved:
        raise ValueError("This solution does not contain /neutral_flux_limiter_diagnostics.")
    recomputed = recompute_neutral_flux_limiter_diagnostics(
        solution,
        atomic_parameters=atomic_parameters,
        neutral_diffusion_parameters=neutral_diffusion_parameters,
        flux_convention=flux_convention,
    )
    if fields is None:
        fields = ("Dnn", "Gamma_unlim", "Gamma_max", "activation_ratio", "phi", "D_eff", "Gamma_lim")
    report = {}
    for field in fields:
        if field not in saved:
            raise KeyError(f"Missing saved neutral limiter diagnostic '{field}'.")
        if field not in recomputed:
            raise KeyError(f"Neutral limiter diagnostic '{field}' was not recomputed.")
        report[field] = _compare_arrays(saved[field], recomputed[field], atol, rtol)

    report["algebraic"] = check_saved_diagnostic_identities(solution, atol=atol, rtol=rtol, strict=False)

    if strict:
        failed = [name for name, item in report.items() if name != "algebraic" and not item["passed"]]
        failed.extend(f"algebraic.{name}" for name, item in report["algebraic"].items() if not item["passed"])
        if failed:
            raise AssertionError(f"Neutral flux limiter diagnostic comparison failed for: {', '.join(failed)}")
    return report


def check_saved_diagnostic_identities(solution, atol=1.0e-12, rtol=1.0e-10, strict=False):
    """Check identities that use saved diagnostics only as consistency targets."""
    diagnostics = solution.neutral_flux_limiter_diagnostics
    if not diagnostics:
        raise ValueError("This solution does not contain /neutral_flux_limiter_diagnostics.")
    required = ("Dnn", "phi", "D_eff", "Gamma_unlim", "Gamma_lim")
    missing = [key for key in required if key not in diagnostics]
    if missing:
        raise KeyError(f"Missing saved neutral limiter diagnostics required for identity checks: {missing}")
    report = {
        "D_eff": _compare_arrays(diagnostics["D_eff"], diagnostics["phi"] * diagnostics["Dnn"], atol, rtol),
        "Gamma_lim": _compare_arrays(
            diagnostics["Gamma_lim"],
            diagnostics["phi"] * diagnostics["Gamma_unlim"],
            atol,
            rtol,
        ),
    }
    if strict:
        failed = [name for name, item in report.items() if not item["passed"]]
        if failed:
            raise AssertionError(f"Saved neutral limiter diagnostic identity checks failed for: {', '.join(failed)}")
    return report


def _compare_arrays(reference, candidate, atol, rtol):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        return {
            "reference_shape": reference.shape,
            "candidate_shape": candidate.shape,
            "max_abs_diff": np.inf,
            "max_rel_diff": np.inf,
            "rms_diff": np.inf,
            "worst_index": None,
            "passed": False,
        }

    diff = candidate - reference
    abs_diff = np.abs(diff)
    denominator = np.maximum(np.abs(reference), atol)
    rel_diff = abs_diff / denominator
    worst_flat = int(np.nanargmax(abs_diff)) if abs_diff.size else 0
    worst_index = tuple(int(i) for i in np.unravel_index(worst_flat, reference.shape)) if abs_diff.size else ()
    return {
        "reference_shape": reference.shape,
        "candidate_shape": candidate.shape,
        "max_abs_diff": float(np.nanmax(abs_diff)) if abs_diff.size else 0.0,
        "max_rel_diff": float(np.nanmax(rel_diff)) if rel_diff.size else 0.0,
        "rms_diff": float(np.sqrt(np.nanmean(diff**2))) if diff.size else 0.0,
        "worst_index": worst_index,
        "reference_at_worst": float(reference[worst_index]) if worst_index else 0.0,
        "candidate_at_worst": float(candidate[worst_index]) if worst_index else 0.0,
        "passed": bool(np.allclose(reference, candidate, atol=atol, rtol=rtol, equal_nan=True)),
    }


def _calculate_gamma_max(solution, u, neutral_diffusion_parameters):
    physics = solution.parameters["physics"]
    rho = u[:, :, solution.metadata.indices.conservative[b"rho"]]
    gamma = u[:, :, solution.metadata.indices.conservative[b"Gamma"]]
    n_ei = u[:, :, solution.metadata.indices.conservative[b"nEi"]]
    rhon = u[:, :, solution.metadata.indices.conservative[b"rhon"]]
    mref = _scalar(physics, "Mref")
    ti = 2.0 / 3.0 / mref * (n_ei / rho - 0.5 * gamma**2 / rho**2)
    ti_limited = _limit_ti_adim(ti, neutral_diffusion_parameters, solution)
    gamma_fs = _scalar(physics, "neutral_flux_limiter_fs_fraction")
    gamma_min = _scalar(physics, "neutral_flux_limiter_fs_flux_min", 0.0)
    raw = gamma_fs * np.maximum(rhon, 0.0) * np.sqrt(mref * ti_limited)
    return np.maximum(raw, gamma_min)


def _limit_ti_adim(ti_adim, neutral_diffusion_parameters, solution):
    ti_min = float(neutral_diffusion_parameters.get("ti_min", 1.0e-6))
    ti_min_adim = ti_min / _scalar(solution.parameters["adimensionalization"], "temperature_scale")
    if neutral_diffusion_parameters.get("ti_soft", False):
        from hdg_postprocess.routines.tools import softplus

        return softplus(
            ti_adim,
            ti_min_adim,
            neutral_diffusion_parameters.get("ti_w", 0.01),
            neutral_diffusion_parameters.get("ti_width", 10),
        )
    return np.maximum(ti_adim, ti_min_adim)


def _prepare_neutral_diffusion_parameters(parameters, adimensionalization):
    params = deepcopy(parameters) if parameters is not None else make_neutral_diffusion_parameters()
    diffusion_scale = _scalar(adimensionalization, "length_scale") ** 2 / _scalar(adimensionalization, "time_scale")
    if "dnn_max_adim" not in params and "dnn_max" in params:
        params["dnn_max_adim"] = params["dnn_max"] / diffusion_scale
    if "dnn_min_adim" not in params and "dnn_min" in params:
        params["dnn_min_adim"] = params["dnn_min"] / diffusion_scale
    return params


def _resolve_atomic_parameters(solution, atomic_parameters):
    if atomic_parameters is not None:
        return atomic_parameters
    atomic = solution.parameters["physics"].get("atomic")
    if atomic is not None:
        return atomic
    atomic = solution.additional_parameters.atomic
    if atomic is not None:
        return atomic
    raise ValueError(
        "Atomic parameters are required to independently recompute neutral Dnn with neutral-neutral collisions. "
        "Pass atomic_parameters=... or configure solution.additional_parameters.set_atomic(...)."
    )


def _require_supported_limiter_mode(solution):
    mode = solution.parameters["physics"].get("neutral_flux_limiter_mode")
    if mode is None:
        raise ValueError("Missing physics parameter 'neutral_flux_limiter_mode'.")
    normalized = _decode_scalar(mode)
    if normalized not in SUPPORTED_RECOMPUTE_MODES:
        raise ValueError(
            f"Unsupported neutral_flux_limiter_mode={normalized!r}. "
            f"Supported modes are: {', '.join(SUPPORTED_RECOMPUTE_MODES)}."
        )


def _require_conservative_variables(solution, names):
    missing = [name for name in names if name not in solution.metadata.indices.conservative]
    if missing:
        raise KeyError(f"Missing conservative variables required for neutral limiter verification: {missing}")


def _dataset_name(name):
    reverse = {value: key for key, value in ACCESSOR_NAMES.items()}
    return reverse.get(name, name)


def _scalar(container, key, default=None):
    if key not in container:
        if default is None:
            raise KeyError(f"Missing required parameter '{key}'.")
        return float(default)
    value = np.asarray(container[key])
    if value.size == 0:
        if default is None:
            raise KeyError(f"Parameter '{key}' is empty.")
        return float(default)
    return float(value.reshape(-1)[0])


def _decode_scalar(value):
    array = np.asarray(value).reshape(-1)
    item = array[0]
    if isinstance(item, bytes):
        return item.decode().strip()
    return str(item).strip()


def split_solution_file_path(path):
    """Return loader-style folder and base name for a single HDF5 solution file."""
    file_path = Path(path)
    if file_path.suffix != ".h5":
        raise ValueError(f"Expected an .h5 solution file path, got: {path}")
    return str(file_path.parent) + "/", file_path.stem
