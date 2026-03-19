import json

import numpy as np

from .common import rectangular_grid_metadata, store_node_field
from .config import IMASExportMetadata, RectangularGrid2D
from .evaluate import equilibrium_interpolators, evaluate_interpolators_on_grid
from .ggd_geometry import populate_grid_reference_ggd_entry, populate_rectangular_grid_ggd


def build_equilibrium_ids(
    solution,
    metadata: IMASExportMetadata,
    grid: RectangularGrid2D,
    *,
    grid_reference_path=None,
):
    """Build a rectangular-grid equilibrium IDS from one HDG solution."""

    import imas

    solution.assembly.full()
    solution.assembly.simple()
    solution.equilibrium.define_axis()

    eq = imas.IDSFactory().equilibrium()
    eq.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    eq.time = np.array([float(metadata.time)])
    eq.time_slice.resize(1)

    ts = eq.time_slice[0]
    ts.time = float(metadata.time)
    _populate_equilibrium_grid(eq, grid, metadata.time, grid_reference_path)

    sampled = sample_equilibrium_fields(solution, grid)
    populate_equilibrium_timeslice(ts, sampled, grid_index=1)
    axis = solution.summary.equilibrium.axis
    if axis.r is not None and axis.z is not None:
        ts.global_quantities.magnetic_axis.r = float(axis.r)
        ts.global_quantities.magnetic_axis.z = float(axis.z)

    eq.code.name = "SOLEDGE-HDG"
    eq.code.repository = "hdg_postprocess"
    eq.code.description = "Equilibrium exported from SOLEDGE-HDG by hdg_postprocess."

    eq_metadata = build_equilibrium_metadata(solution, metadata, grid)
    if grid_reference_path is not None:
        eq_metadata["ggd_grid_reference_path"] = str(grid_reference_path)
        eq_metadata["ggd_grid_reference_note"] = (
            "equilibrium.grids_ggd references the topology stored in plasma_profiles.grid_ggd "
            "to avoid duplicate rectangular GGD geometry."
        )
    eq.code.parameters = json.dumps(eq_metadata, sort_keys=True)

    return eq


def put_equilibrium(entry, solution, metadata: IMASExportMetadata, grid: RectangularGrid2D):
    """Build and store one equilibrium IDS in the provided IMAS DBEntry."""

    equilibrium = build_equilibrium_ids(solution, metadata, grid)
    entry.put(equilibrium, metadata.occurrence)
    return equilibrium


def _plasma_grid_reference_path(*, occurrence, grid_index):
    return f"#plasma_profiles:{int(occurrence)}/grid_ggd({int(grid_index)})"


def _populate_equilibrium_grid(eq, grid, time_value, grid_reference_path):
    if grid_reference_path is None:
        populate_rectangular_grid_ggd(eq, grid, float(time_value))
        return

    eq.grids_ggd.resize(1)
    populate_grid_reference_ggd_entry(
        eq.grids_ggd[0],
        time_value=float(time_value),
        path=str(grid_reference_path),
        grid_name="rectangular_rz",
        grid_index=1,
    )


def sample_equilibrium_fields(solution, grid):
    r_grid, z_grid = grid.mesh()
    interpolators = equilibrium_interpolators(solution)

    eval_fields = {
        "psi": interpolators["psi"],
        "br": interpolators["br"],
        "bz": interpolators["bz"],
        "bphi": interpolators["bphi"],
    }
    if interpolators["jphi"] is not None:
        eval_fields["jphi"] = interpolators["jphi"]

    sampled = evaluate_interpolators_on_grid(
        eval_fields,
        r_grid=r_grid,
        z_grid=z_grid,
        locator=solution.mesh.geometry.element_locator,
        outside_value=np.nan,
    )

    psi_values = sampled["psi"].reshape(-1)
    finite_psi = psi_values[np.isfinite(psi_values)]
    return {
        "psi": psi_values,
        "b_field_r": sampled["br"].reshape(-1),
        "b_field_z": sampled["bz"].reshape(-1),
        "b_field_phi": sampled["bphi"].reshape(-1),
        "j_phi": sampled["jphi"].reshape(-1) if "jphi" in sampled else None,
        "psi_axis": float(np.nanmin(finite_psi)) if finite_psi.size else None,
    }


def populate_equilibrium_timeslice(ts, sampled, *, grid_index):
    ts.ggd.resize(1)
    ggd = ts.ggd[0]

    store_node_field(ggd.psi, sampled["psi"], grid_index=grid_index, flatten=False)
    store_node_field(ggd.b_field_r, sampled["b_field_r"], grid_index=grid_index, flatten=False)
    store_node_field(ggd.b_field_z, sampled["b_field_z"], grid_index=grid_index, flatten=False)
    store_node_field(ggd.b_field_phi, sampled["b_field_phi"], grid_index=grid_index, flatten=False)
    if sampled["j_phi"] is not None:
        store_node_field(ggd.j_phi, sampled["j_phi"], grid_index=grid_index, flatten=False)

    if sampled["psi_axis"] is not None:
        ts.global_quantities.psi_axis = sampled["psi_axis"]

    if sampled.get("axis_r") is not None and sampled.get("axis_z") is not None:
        ts.global_quantities.magnetic_axis.r = sampled["axis_r"]
        ts.global_quantities.magnetic_axis.z = sampled["axis_z"]


def build_equilibrium_metadata(solution, metadata, grid):
    extracted = rectangular_grid_metadata(grid)
    extracted.update({
        "representation_note": "Exported on a rectangular cylindrical (R,Z) mesh through equilibrium.grids_ggd/time_slice[0].ggd.",
        "ggd_grid_description": "Regular cylindrical (R,Z) rectangular grid with explicit node, edge, and cell topology.",
        "ggd_grid_subset": "All exported equilibrium fields currently live on the nodes subset.",
        "ggd_value_ordering": extracted["value_ordering"],
        "poloidal_flux_export_note": (
            metadata.poloidal_flux_convention
            or (
                "Poloidal flux is exported exactly as stored in the HDG solution; "
                "normalization and sign may depend on solver version, so the person "
                "doing the IMAS export should verify the convention."
            )
        ),
        "j_phi_export_note": (
            "Exported j_phi currently comes from the HDG Jtor field; in current usage "
            "this is typically the Ohmic contribution, and the person doing the IMAS "
            "export should verify the sign convention."
        ),
    })
    extracted.pop("value_ordering")

    axis = solution.summary.equilibrium.axis
    if axis.r is not None and axis.z is not None:
        extracted["magnetic_axis_r_m"] = float(axis.r)
        extracted["magnetic_axis_z_m"] = float(axis.z)
    return extracted
