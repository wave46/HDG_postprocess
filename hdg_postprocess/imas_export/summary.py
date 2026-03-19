import json

import numpy as np

from .common import extend_time_metadata
from .config import IMASExportMetadata


_TRANSPORT_KEYS = (
    "diff_n",
    "diff_u",
    "diff_e",
    "diff_ee",
    "diff_nn",
    "diff_pare",
    "diff_pari",
)


def build_summary_ids(solution, metadata: IMASExportMetadata):
    """Build a summary IDS from one HDG solution and export metadata."""

    import imas

    summary = imas.IDSFactory().summary()
    extracted = summary_export_metadata(solution, metadata)
    populate_summary_ids(
        summary,
        metadata,
        extracted,
        time_values=[float(metadata.time)],
        workflow="steady_state" if extracted.get("steady_state", False) else "time_dependent_snapshot",
        description_prefix="Exported from SOLEDGE-HDG by hdg_postprocess.",
    )

    return summary


def put_summary(entry, solution, metadata: IMASExportMetadata):
    """Build and store one summary IDS in the provided IMAS DBEntry."""

    summary = build_summary_ids(solution, metadata)
    entry.put(summary, metadata.occurrence)
    return summary


def extract_solution_summary_metadata(solution):
    """Extract a compact run-metadata mapping from the loaded SOLEDGE-HDG solution."""

    parameters = solution.parameters
    physics = parameters.get("physics", {})
    switches = parameters.get("switches", {})
    extracted = {}

    mapping = {
        "puff_rate": physics.get("puff"),
        "recycling_coefficient": physics.get("recycling"),
        "pump_recycling_coefficient": physics.get("recycling_pump"),
        "impurity_name": physics.get("impurity_name"),
        "impurity_concentration": physics.get("impurity_concentration"),
        "Gmbohm": physics.get("Gmbohm"),
        "Gmbohme": physics.get("Gmbohme"),
        "bohmth": physics.get("bohmth"),
        "ohmic_coeff": physics.get("ohmic_coeff"),
        "Zeff": physics.get("Zeff"),
    }
    for key, value in mapping.items():
        if value is None:
            continue
        extracted[key] = _decode_if_bytes(_as_python_scalar(value))

    extracted.update(_extract_transport_metadata(parameters))

    if "steady" in switches:
        extracted["steady_state"] = bool(_as_python_scalar(switches["steady"]))
    if "impurity_radiation" in switches:
        extracted["impurity_radiation_enabled"] = bool(_as_python_scalar(switches["impurity_radiation"]))
    if "ohmicsrc" in switches:
        extracted["ohmic_source_enabled"] = bool(_as_python_scalar(switches["ohmicsrc"]))
    if "testcase" in switches:
        extracted["testcase"] = int(_as_python_scalar(switches["testcase"]))

    if "bohm_energy_thresh" in physics:
        extracted["bohm_energy_thresh"] = _as_python_scalar(physics["bohm_energy_thresh"])

    return extracted


def summary_export_metadata(solution, metadata: IMASExportMetadata, *, times=None):
    extracted = extract_solution_summary_metadata(solution)
    extracted["export_shot"] = int(metadata.shot)
    extracted["export_run"] = int(metadata.run)
    extracted["export_occurrence"] = int(metadata.occurrence)
    extracted["effective_energy_transfer"] = float(metadata.effective_energy_transfer)
    if times is not None:
        extend_time_metadata(extracted, [float(time_value) for time_value in times])
    return extracted


def populate_summary_ids(
    summary,
    metadata: IMASExportMetadata,
    extracted,
    *,
    time_values,
    workflow,
    description_prefix,
):
    import imas

    summary.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    summary.description = metadata.description
    summary.time = np.asarray(time_values, dtype=float)
    summary.pulse = int(metadata.shot)

    if metadata.machine:
        summary.machine = metadata.machine

    summary.ids_properties.comment = _build_ids_comment(metadata, extracted)
    if metadata.comment:
        summary.tag.comment = metadata.comment

    summary.code.name = "SOLEDGE-HDG"
    summary.code.repository = "hdg_postprocess"
    summary.code.description = _summary_code_description(description_prefix, extracted)
    summary.code.parameters = json.dumps(extracted, sort_keys=True)
    summary.simulation.workflow = str(workflow)

    puff_rate = extracted.get("puff_rate")
    if puff_rate is not None:
        summary.gas_injection_rates.total.value = np.full(len(time_values), float(puff_rate), dtype=float)
        summary.gas_injection_rates.total.source = "SOLEDGE-HDG physics/puff"


def _build_ids_comment(metadata, extracted):
    lines = []
    if metadata.comment:
        lines.append(metadata.comment)
    if extracted:
        lines.append("Simulation metadata is stored in summary.code.parameters as JSON.")
    return "\n".join(lines)


def _extract_transport_metadata(parameters):
    physics = parameters.get("physics", {})
    extracted = {}

    for key in _TRANSPORT_KEYS:
        source_key = _transport_source_key(physics, key)
        if source_key not in physics:
            continue
        if key in ("diff_pare", "diff_pari"):
            extracted[f"{key}_m2_s"], source_key = _parallel_conductivity_value(parameters, key)
        else:
            extracted[f"{key}_m2_s"] = _scaled_value(parameters, "physics", source_key, "diffusion_scale")
        if source_key != key:
            extracted[f"{key}_source_key"] = source_key

    return extracted


def _parallel_conductivity_value(parameters, key):
    physics = parameters.get("physics", {})
    adim = parameters.get("adimensionalization", {})
    source_key = _transport_source_key(physics, key)
    if source_key is None:
        return None, None

    denom = (
        _as_python_scalar(adim["time_scale"]) ** 3
        * _as_python_scalar(adim["temperature_scale"]) ** (7 / 2)
        / (
            _as_python_scalar(adim["density_scale"])
            * _as_python_scalar(adim["length_scale"]) ** 4
        )
        / _as_python_scalar(adim["mass_scale"])
    )
    return _as_python_scalar(physics[source_key]) / denom, source_key


def _transport_source_key(physics, key):
    for candidate in (f"ME_{key}", f"{key}_ME", key):
        if candidate in physics:
            return candidate
    return None


def _scaled_value(parameters, group_name, key, scale_key):
    group = parameters.get(group_name, {})
    adim = parameters.get("adimensionalization", {})
    if key not in group or scale_key not in adim:
        return None
    return _as_python_scalar(group[key]) * _as_python_scalar(adim[scale_key])


def _decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def _as_python_scalar(value):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summary_code_description(description_prefix, extracted):
    testcase = extracted.get("testcase")
    if testcase is None:
        return description_prefix
    return f"{description_prefix.rstrip('.')} (testcase {testcase})."
