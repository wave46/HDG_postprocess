import json

import numpy as np

from .config import IMASExportMetadata, RectangularGrid2D
from .evaluate import equilibrium_interpolators, evaluate_interpolators_on_grid
from .ggd_geometry import populate_grid_reference_ggd_entry, populate_rectangular_grid_ggd


def _equilibrium_metadata(solution, metadata, grid):
    extracted = {
        "grid_shape": [grid.nr, grid.nz],
        "grid_r_range_m": [grid.r_min, grid.r_max],
        "grid_z_range_m": [grid.z_min, grid.z_max],
        "representation_note": "Exported on a rectangular cylindrical (R,Z) mesh through equilibrium.grids_ggd/time_slice[0].ggd.",
        "ggd_grid_name": "rectangular_rz",
        "ggd_grid_description": "Regular cylindrical (R,Z) rectangular grid with explicit node, edge, and cell topology.",
        "ggd_grid_subset": "All exported equilibrium fields currently live on the nodes subset.",
        "ggd_value_ordering": "Node values are flattened from meshgrid(indexing='ij') in C order, so R is the slow axis and Z the fast axis.",
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
    }
    if solution.summary.equilibrium.axis.r is not None and solution.summary.equilibrium.axis.z is not None:
        extracted["magnetic_axis_r_m"] = float(solution.summary.equilibrium.axis.r)
        extracted["magnetic_axis_z_m"] = float(solution.summary.equilibrium.axis.z)
    return extracted


def _plasma_grid_reference_path(*, occurrence, grid_index):
    return f"#plasma_profiles:{int(occurrence)}/grid_ggd({int(grid_index)})"


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
    if grid_reference_path is None:
        populate_rectangular_grid_ggd(eq, grid, float(metadata.time))
    else:
        eq.grids_ggd.resize(1)
        populate_grid_reference_ggd_entry(
            eq.grids_ggd[0],
            time_value=float(metadata.time),
            path=str(grid_reference_path),
            grid_name="rectangular_rz",
            grid_index=1,
        )
    ts.ggd.resize(1)
    ggd = ts.ggd[0]

    r_grid, z_grid = grid.mesh()
    interpolators = equilibrium_interpolators(solution)
    locator = solution.mesh.geometry.element_locator
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
        locator=locator,
        outside_value=np.nan,
    )
    psi_values = sampled["psi"].reshape(-1)
    br_values = sampled["br"].reshape(-1)
    bz_values = sampled["bz"].reshape(-1)
    bphi_values = sampled["bphi"].reshape(-1)
    def _store_ggd_field(field_name, values):
        field = getattr(ggd, field_name)
        field.resize(1)
        field[0].grid_index = 1
        field[0].grid_subset_index = 1
        field[0].values = values

    _store_ggd_field("psi", psi_values)
    _store_ggd_field("b_field_r", br_values)
    _store_ggd_field("b_field_z", bz_values)
    _store_ggd_field("b_field_phi", bphi_values)
    if interpolators["jphi"] is not None:
        _store_ggd_field("j_phi", sampled["jphi"].reshape(-1))

    axis = solution.summary.equilibrium.axis
    if axis.r is not None and axis.z is not None:
        ts.global_quantities.magnetic_axis.r = float(axis.r)
        ts.global_quantities.magnetic_axis.z = float(axis.z)

    finite_psi = psi_values[np.isfinite(psi_values)]
    if finite_psi.size:
        ts.global_quantities.psi_axis = float(np.nanmin(finite_psi))

    eq.code.name = "SOLEDGE-HDG"
    eq.code.repository = "hdg_postprocess"
    eq.code.description = "Equilibrium exported from SOLEDGE-HDG by hdg_postprocess."
    eq_metadata = _equilibrium_metadata(solution, metadata, grid)
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
