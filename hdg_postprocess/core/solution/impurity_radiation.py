from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_COEFFICIENT_COUNT = 17


@dataclass(frozen=True)
class ImpurityRadiationMetadata:
    n_impurities: int
    impurity_names: tuple[str, ...]
    impurity_concentrations: np.ndarray
    impurity_cooling_coefficients_adim: np.ndarray | None


def get_impurity_radiation_metadata(solution, *, require_coefficients=False):
    """Return normalized impurity-radiation metadata, or None when unavailable."""
    physics = solution.parameters.get("physics", {})
    atomic = getattr(getattr(solution, "additional_parameters", None), "atomic", None)
    return normalize_impurity_radiation_metadata(
        physics,
        atomic_parameters=atomic,
        require_coefficients=require_coefficients,
    )


def normalize_impurity_radiation_metadata(
    physics,
    *,
    atomic_parameters=None,
    require_coefficients=False,
):
    """Normalize new mixture metadata and legacy scalar impurity settings.

    New Fortran outputs store plural impurity metadata directly in ``physics``.
    Older files store only scalar ``impurity_name`` / ``impurity_concentration``;
    when old atomic setup has been configured, its single cooling table is exposed
    as a one-column coefficient matrix.
    """
    if _has_new_mixture_metadata(physics):
        return _metadata_from_mixture_fields(physics, require_coefficients=require_coefficients)
    if _has_legacy_scalar_metadata(physics):
        return _metadata_from_legacy_fields(
            physics,
            atomic_parameters=atomic_parameters,
            require_coefficients=require_coefficients,
        )
    if require_coefficients:
        raise ValueError("Impurity radiation metadata is not available.")
    return None


def has_impurity_radiation(solution):
    metadata = get_impurity_radiation_metadata(solution, require_coefficients=False)
    return metadata is not None and np.any(metadata.impurity_concentrations > 0.0)


def _has_new_mixture_metadata(physics):
    return (
        "n_impurities" in physics
        or "impurity_names" in physics
        or "impurity_concentrations" in physics
        or "impurity_cooling_coefficients_adim" in physics
    )


def _has_legacy_scalar_metadata(physics):
    return "impurity_name" in physics or "impurity_concentration" in physics


def _metadata_from_mixture_fields(physics, *, require_coefficients):
    missing = [
        key
        for key in ("n_impurities", "impurity_names", "impurity_concentrations")
        if key not in physics
    ]
    if missing:
        raise ValueError(f"Incomplete impurity radiation metadata; missing {missing}.")

    n_impurities = int(_as_scalar(physics["n_impurities"]))
    names = _normalize_names(physics["impurity_names"], n_impurities)
    concentrations = _normalize_concentrations(physics["impurity_concentrations"], n_impurities)
    coefficients = None
    if "impurity_cooling_coefficients_adim" in physics:
        coefficients = _normalize_coefficients(
            physics["impurity_cooling_coefficients_adim"],
            n_impurities,
        )
    elif require_coefficients:
        raise ValueError("physics/impurity_cooling_coefficients_adim is required.")

    return ImpurityRadiationMetadata(
        n_impurities=n_impurities,
        impurity_names=names,
        impurity_concentrations=concentrations,
        impurity_cooling_coefficients_adim=coefficients,
    )


def _metadata_from_legacy_fields(physics, *, atomic_parameters, require_coefficients):
    name = _decode_scalar(physics.get("impurity_name", ""))
    concentration = float(_as_scalar(physics.get("impurity_concentration", 0.0)))
    coefficients = _legacy_cooling_coefficients(atomic_parameters)
    if coefficients is None and require_coefficients:
        raise ValueError(
            "Legacy impurity metadata needs additional_parameters.atomic['cooling_factor']['alpha'] "
            "to provide cooling coefficients."
        )
    if coefficients is not None:
        coefficients = _normalize_coefficients(coefficients, 1)
    return ImpurityRadiationMetadata(
        n_impurities=1,
        impurity_names=(name,),
        impurity_concentrations=np.asarray([concentration], dtype=float),
        impurity_cooling_coefficients_adim=coefficients,
    )


def _legacy_cooling_coefficients(atomic_parameters):
    if atomic_parameters is None or "cooling_factor" not in atomic_parameters:
        return None
    cooling = atomic_parameters["cooling_factor"]
    if cooling is None or "alpha" not in cooling:
        return None
    return cooling["alpha"]


def _normalize_names(values, n_impurities):
    raw = np.asarray(values)
    if raw.ndim == 0:
        names = [_decode_scalar(raw.item())]
    elif raw.dtype.kind in {"S", "U", "O"}:
        names = [_decode_scalar(item) for item in raw.reshape(-1)]
    elif raw.dtype.kind in {"i", "u"} and raw.ndim >= 2:
        # Some HDF5 string datasets arrive as a fixed-width character code matrix.
        names = [_decode_char_codes(row) for row in _name_rows(raw, n_impurities)]
    else:
        names = [_decode_scalar(item) for item in raw.reshape(-1)]

    if len(names) != n_impurities:
        raise ValueError(
            f"Expected {n_impurities} impurity names, got {len(names)}."
        )
    return tuple(names)


def _name_rows(raw, n_impurities):
    if raw.shape[0] == n_impurities:
        return raw
    if raw.shape[-1] == n_impurities:
        return raw.T
    return raw.reshape(n_impurities, -1)


def _normalize_concentrations(values, n_impurities):
    concentrations = np.asarray(values, dtype=float).reshape(-1)
    if concentrations.size != n_impurities:
        raise ValueError(
            f"Expected {n_impurities} impurity concentrations, got {concentrations.size}."
        )
    return concentrations


def _normalize_coefficients(values, n_impurities):
    coefficients = np.asarray(values, dtype=float)
    if coefficients.ndim == 1 and n_impurities == 1:
        coefficients = coefficients[:, None]
    elif coefficients.shape == (n_impurities, _COEFFICIENT_COUNT):
        coefficients = coefficients.T
    elif coefficients.shape != (_COEFFICIENT_COUNT, n_impurities):
        raise ValueError(
            "Expected impurity cooling coefficients with shape "
            f"({_COEFFICIENT_COUNT}, {n_impurities}) or ({n_impurities}, {_COEFFICIENT_COUNT}); "
            f"got {coefficients.shape}."
        )

    if coefficients.shape != (_COEFFICIENT_COUNT, n_impurities):
        raise ValueError(
            f"Expected normalized coefficient shape ({_COEFFICIENT_COUNT}, {n_impurities}); "
            f"got {coefficients.shape}."
        )
    return coefficients


def _as_scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar value, got shape {array.shape}.")
    return array.reshape(-1)[0]


def _decode_scalar(value):
    value = _as_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


def _decode_char_codes(values):
    chars = []
    for value in np.asarray(values).reshape(-1):
        code = int(value)
        if code == 0:
            continue
        chars.append(chr(code))
    return "".join(chars).strip()
