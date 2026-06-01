from pathlib import Path

import h5py
import numpy as np

from hdg_postprocess.core.solution.neutral_flux_limiter import split_solution_file_path


DIAGNOSTIC_GROUP = "diagnostics/balance/neutrals/particles/nodal_wall_sources"
LEGACY_DIAGNOSTIC_GROUP = "neutral_wall_sources_diagnostics"
FLUX_DENSITY_FIELDS = ("puff_flux_density", "pump_flux_density", "net_flux_density")
TOTAL_FIELDS = ("element_puff_total", "element_pump_total", "element_net_total")
SUPPORTED_FIELDS = (*FLUX_DENSITY_FIELDS, *TOTAL_FIELDS)


def read_neutral_wall_source_diagnostics(path):
    """Read neutral wall-source diagnostics from one HDF5 solution file."""
    with h5py.File(path, "r") as h5:
        if DIAGNOSTIC_GROUP in h5:
            group = h5[DIAGNOSTIC_GROUP]
        elif LEGACY_DIAGNOSTIC_GROUP in h5:
            group = h5[LEGACY_DIAGNOSTIC_GROUP]
        else:
            raise ValueError(
                f"File does not contain /{DIAGNOSTIC_GROUP} or /{LEGACY_DIAGNOSTIC_GROUP}: {path}"
            )
        if "mesh/Nelems" not in h5 or "mesh/Nnodesperelem" not in h5:
            raise KeyError("Neutral wall-source diagnostics require /mesh/Nelems and /mesh/Nnodesperelem.")
        n_elements = int(np.asarray(h5["mesh/Nelems"]).reshape(-1)[0])
        nodes_per_element = int(np.asarray(h5["mesh/Nnodesperelem"]).reshape(-1)[0])
        diagnostics = {}
        for key, dataset in group.items():
            if key not in SUPPORTED_FIELDS:
                continue
            values = dataset[()]
            if key in TOTAL_FIELDS:
                diagnostics[key] = float(np.asarray(values).reshape(-1)[0])
            else:
                diagnostics[key] = _reshape_flux_density(values, n_elements, nodes_per_element, key)
    return diagnostics


def collect_neutral_wall_source_totals(paths):
    """Return integrated neutral wall-source totals for a sequence of solution files."""
    rows = []
    for path in paths:
        diagnostics = read_neutral_wall_source_diagnostics(path)
        rows.append({"path": str(path), **{key: diagnostics.get(key, np.nan) for key in TOTAL_FIELDS}})
    return rows


def neutral_wall_source_field(solution, field="net_flux_density"):
    """Return a loaded wall-source flux-density field from an HDGsolution."""
    if field not in FLUX_DENSITY_FIELDS:
        raise ValueError(f"Unsupported neutral wall-source field '{field}'. Supported fields: {FLUX_DENSITY_FIELDS}")
    diagnostics = solution.neutral_wall_source_diagnostics
    if not diagnostics:
        raise ValueError(f"This solution does not contain /{DIAGNOSTIC_GROUP} or /{LEGACY_DIAGNOSTIC_GROUP}.")
    if field not in diagnostics:
        raise KeyError(f"Neutral wall-source diagnostic '{field}' is not available.")
    return diagnostics[field]


def neutral_wall_source_totals(solution):
    """Return integrated neutral wall-source totals from an HDGsolution."""
    diagnostics = solution.neutral_wall_source_diagnostics
    if not diagnostics:
        raise ValueError(f"This solution does not contain /{DIAGNOSTIC_GROUP} or /{LEGACY_DIAGNOSTIC_GROUP}.")
    return {key: diagnostics.get(key, np.nan) for key in TOTAL_FIELDS}


def check_neutral_wall_source_identities(solution=None, diagnostics=None, atol=1.0e-12, rtol=1.0e-10):
    """Check saved neutral wall-source net density and total identities."""
    if diagnostics is None:
        if solution is None:
            raise ValueError("Provide either solution=... or diagnostics=....")
        diagnostics = solution.neutral_wall_source_diagnostics
    missing = [key for key in (*FLUX_DENSITY_FIELDS, *TOTAL_FIELDS) if key not in diagnostics]
    if missing:
        raise KeyError(f"Missing neutral wall-source diagnostics required for identity checks: {missing}")

    density_expected = diagnostics["puff_flux_density"] - diagnostics["pump_flux_density"]
    total_expected = diagnostics["element_puff_total"] - diagnostics["element_pump_total"]
    return {
        "net_flux_density": _compare_arrays(diagnostics["net_flux_density"], density_expected, atol, rtol),
        "element_net_total": _compare_arrays(
            np.asarray([diagnostics["element_net_total"]]),
            np.asarray([total_expected]),
            atol,
            rtol,
        ),
    }


def plot_neutral_wall_source(path, field="net_flux_density", ax=None, log=False, cmap=None, limits=None):
    """Plot one wall-source flux-density diagnostic as a discontinuous element-node field."""
    from hdg_postprocess.formats.load_from_file import load_HDG_solution_from_file

    if field not in FLUX_DENSITY_FIELDS:
        raise ValueError(f"Unsupported neutral wall-source field '{field}'. Supported fields: {FLUX_DENSITY_FIELDS}")
    solution_path, solution_base = split_solution_file_path(path)
    solution = load_HDG_solution_from_file(solution_path, solution_base)
    values = neutral_wall_source_field(solution, field)
    return plot_element_node_field(solution, values, ax=ax, log=log, cmap=cmap, limits=limits, label=field)


def plot_element_node_field(solution, values, ax=None, log=False, cmap=None, limits=None, label=None):
    """Plot a discontinuous element-node field by duplicating mesh nodes per element."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.tri import Triangulation

    if not solution.mesh.metadata.flags.combined_to_full:
        solution.mesh.assembly.full()
    values = np.asarray(values)
    connectivity = solution.mesh.global_state.connectivity
    if values.shape != connectivity.shape:
        raise ValueError(f"Field shape {values.shape} does not match element-node connectivity {connectivity.shape}.")

    local_tris = _local_element_triangulation(solution.mesh)
    n_elements, nodes_per_element = connectivity.shape
    vertices = solution.mesh.global_state.vertices[connectivity].reshape(n_elements * nodes_per_element, 2)
    offsets = np.arange(n_elements)[:, None, None] * nodes_per_element
    triangles = (local_tris[None, :, :] + offsets).reshape(n_elements * local_tris.shape[0], 3)
    plot_values = values.reshape(-1)
    triangulation = Triangulation(vertices[:, 0], vertices[:, 1], triangles=triangles)

    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    norm = _plot_norm(plot_values, log=log, limits=limits)
    if cmap is None:
        cmap = "magma"
    artist = ax.tripcolor(triangulation, plot_values, shading="gouraud", cmap=cmap, norm=norm)
    if limits is not None and not log:
        artist.set_clim(limits[0], limits[1])
    cbar = plt.colorbar(artist, ax=ax)
    if label is not None:
        cbar.set_label(label)
        ax.set_title(label if not log else f"log10({label})")
    ax.set_aspect(1)
    ax.set_xlim(solution.mesh.metadata.extent["minr"], solution.mesh.metadata.extent["maxr"])
    ax.set_ylim(solution.mesh.metadata.extent["minz"], solution.mesh.metadata.extent["maxz"])
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    return ax, artist


def _local_element_triangulation(mesh):
    if mesh.metadata.p_order is None:
        if mesh.mesh_parameters["element_type"] == "triangle":
            return np.asarray([[0, 1, 2]], dtype=int)
        if mesh.mesh_parameters["element_type"] == "quadrilateral":
            return np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "triangulations_element"
        / f"{mesh.mesh_parameters['element_type']}_P{mesh.metadata.p_order}.npy"
    )
    if not path.exists():
        raise KeyError(f"{path} splitting of each element is not defined yet")
    return np.load(path)


def _reshape_flux_density(values, n_elements, nodes_per_element, key):
    array = np.asarray(values)
    expected_size = n_elements * nodes_per_element
    if array.size != expected_size:
        raise ValueError(
            f"Neutral wall-source diagnostic '{key}' has flat size {array.size}, "
            f"expected {expected_size} = {n_elements} * {nodes_per_element}."
        )
    return array.reshape(n_elements, nodes_per_element)


def _plot_norm(values, log=False, limits=None):
    from matplotlib.colors import LogNorm, Normalize

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    if log:
        if limits is not None:
            return LogNorm(vmin=10.0 ** limits[0], vmax=10.0 ** limits[1])
        positive = finite[finite > 0]
        if positive.size == 0:
            return None
        return LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    if limits is not None:
        return Normalize(vmin=limits[0], vmax=limits[1])
    return Normalize(vmin=float(finite.min()), vmax=float(finite.max()))


def _compare_arrays(reference, candidate, atol, rtol):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    diff = candidate - reference
    abs_diff = np.abs(diff)
    denominator = np.maximum(np.abs(reference), atol)
    rel_diff = abs_diff / denominator
    return {
        "max_abs_diff": float(np.nanmax(abs_diff)) if abs_diff.size else 0.0,
        "max_rel_diff": float(np.nanmax(rel_diff)) if rel_diff.size else 0.0,
        "passed": bool(np.allclose(reference, candidate, atol=atol, rtol=rtol, equal_nan=True)),
    }
