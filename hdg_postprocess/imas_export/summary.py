import json

import numpy as np

from .config import IMASExportMetadata


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


def _decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def extract_solution_summary_metadata(solution):
    """Extract a compact run-metadata mapping from the loaded SOLEDGE-HDG solution."""

    physics = solution.parameters.get("physics", {})
    switches = solution.parameters.get("switches", {})
    extracted = {}

    mapping = {
        "puff_rate": physics.get("puff"),
        "recycling_coefficient": physics.get("recycling"),
        "pump_recycling_coefficient": physics.get("recycling_pump"),
        "impurity_name": physics.get("impurity_name"),
        "impurity_concentration": physics.get("impurity_concentration"),
        "testcase": switches.get("testcase"),
    }
    for key, value in mapping.items():
        if value is None:
            continue
        extracted[key] = _decode_if_bytes(_as_python_scalar(value))

    if "steady" in switches:
        extracted["steady_state"] = bool(_as_python_scalar(switches["steady"]))
    if "impurity_radiation" in switches:
        extracted["impurity_radiation_enabled"] = bool(_as_python_scalar(switches["impurity_radiation"]))
    if "ohmicsrc" in switches:
        extracted["ohmic_source_enabled"] = bool(_as_python_scalar(switches["ohmicsrc"]))

    return extracted


def _build_ids_comment(metadata, extracted):
    lines = []
    if metadata.comment:
        lines.append(metadata.comment)
    if extracted:
        lines.append("Simulation metadata is stored in summary.code.parameters as JSON.")
    return "\n".join(lines)


def build_summary_ids(solution, metadata: IMASExportMetadata):
    """Build a summary IDS from one HDG solution and export metadata."""

    import imas

    summary = imas.IDSFactory().summary()
    summary.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    summary.description = metadata.description
    summary.time = [float(metadata.time)]
    summary.pulse = int(metadata.shot)

    if metadata.case_name:
        summary.tag.name = metadata.case_name
    if metadata.machine:
        summary.machine = metadata.machine

    extracted = extract_solution_summary_metadata(solution)
    summary.ids_properties.comment = _build_ids_comment(metadata, extracted)
    if metadata.comment:
        summary.tag.comment = metadata.comment

    summary.code.name = "SOLEDGE-HDG"
    summary.code.repository = "hdg_postprocess"
    summary.code.description = "Exported from SOLEDGE-HDG by hdg_postprocess."
    summary.code.parameters = json.dumps(extracted, sort_keys=True)

    summary.simulation.workflow = (
        "steady_state" if extracted.get("steady_state", False) else "time_dependent_snapshot"
    )

    puff_rate = extracted.get("puff_rate")
    if puff_rate is not None:
        summary.gas_injection_rates.total.value = [float(puff_rate)]
        summary.gas_injection_rates.total.source = "SOLEDGE-HDG physics/puff"

    return summary


def put_summary(entry, solution, metadata: IMASExportMetadata):
    """Build and store one summary IDS in the provided IMAS DBEntry."""

    summary = build_summary_ids(solution, metadata)
    entry.put(summary, metadata.occurrence)
    return summary
