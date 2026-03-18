import json

import numpy as np

from .config import IMASExportMetadata, RectangularGrid2D
from .evaluate import evaluate_variables_on_grid
from .ggd_geometry import populate_rectangular_grid_ggd_array


def _plasma_profiles_metadata(solution, grid):
    physics = solution.parameters["physics"]
    extracted = {
        "grid_shape": [grid.nr, grid.nz],
        "grid_r_range_m": [grid.r_min, grid.r_max],
        "grid_z_range_m": [grid.z_min, grid.z_max],
        "ggd_grid_name": "rectangular_rz",
        "ggd_grid_subset": "All exported plasma fields currently live on the nodes subset.",
        "value_ordering": "Node values are flattened from meshgrid(indexing='ij') in C order, so R is the slow axis and Z the fast axis.",
        "outside_mesh_policy": "Values outside the HDG mesh are exported as NaN.",
        "model_note": "SOLEDGE-HDG currently uses shared plasma density and parallel velocity for electrons and the single ion species.",
    }
    if "Zeff" in physics:
        extracted["Zeff"] = float(physics["Zeff"])
    if "impurity_name" in physics:
        impurity_name = physics["impurity_name"]
        if isinstance(impurity_name, bytes):
            impurity_name = impurity_name.decode()
        elif hasattr(impurity_name, "item"):
            impurity_name = impurity_name.item()
            if isinstance(impurity_name, bytes):
                impurity_name = impurity_name.decode()
        extracted["impurity_name"] = impurity_name
    return extracted


def build_plasma_profiles_ids(solution, metadata: IMASExportMetadata, grid: RectangularGrid2D):
    """Build a rectangular-GGD plasma_profiles IDS from one HDG solution."""

    import imas

    solution.assembly.full()
    solution.assembly.simple()

    plasma = imas.IDSFactory().plasma_profiles()
    plasma.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    plasma.time = np.array([float(metadata.time)])
    populate_rectangular_grid_ggd_array(plasma.grid_ggd, grid, float(metadata.time))
    plasma.ggd.resize(1)

    ggd = plasma.ggd[0]
    ggd.time = float(metadata.time)

    r_grid, z_grid = grid.mesh()
    locator = solution.mesh.geometry.element_locator
    sampled = evaluate_variables_on_grid(
        solution,
        r_grid,
        z_grid,
        ["n", "te", "ti", "u", "nn", "psi"],
        locator=locator,
        outside_value=np.nan,
    )

    def _store_struct_field(field_container, values):
        field_container.resize(1)
        field_container[0].grid_index = 1
        field_container[0].grid_subset_index = 1
        field_container[0].values = values.reshape(-1)

    _store_struct_field(ggd.electrons.density, sampled["n"])
    _store_struct_field(ggd.electrons.temperature, sampled["te"])

    ggd.ion.resize(1)
    ion = ggd.ion[0]
    ion.name = "D+"
    ion.z_ion = 1.0
    _store_struct_field(ion.density, sampled["n"])
    _store_struct_field(ion.temperature, sampled["ti"])
    ion.velocity.resize(1)
    ion.velocity[0].grid_index = 1
    ion.velocity[0].grid_subset_index = 1
    ion.velocity[0].parallel = sampled["u"].reshape(-1)

    ggd.neutral.resize(1)
    neutral = ggd.neutral[0]
    neutral.name = "D"
    _store_struct_field(neutral.density, sampled["nn"])

    _store_struct_field(ggd.n_i_total, sampled["n"])
    _store_struct_field(ggd.t_i_average, sampled["ti"])
    _store_struct_field(ggd.psi, sampled["psi"])
    if "Zeff" in solution.parameters["physics"]:
        zeff_values = np.full(r_grid.shape, float(solution.parameters["physics"]["Zeff"]), dtype=float)
        zeff_values[np.isnan(sampled["n"])] = np.nan
        _store_struct_field(ggd.zeff, zeff_values)

    plasma.code.name = "SOLEDGE-HDG"
    plasma.code.repository = "hdg_postprocess"
    plasma.code.description = "Plasma profiles exported from SOLEDGE-HDG by hdg_postprocess."
    plasma.code.parameters = json.dumps(_plasma_profiles_metadata(solution, grid), sort_keys=True)
    return plasma


def put_plasma_profiles(entry, solution, metadata: IMASExportMetadata, grid: RectangularGrid2D):
    """Build and store one plasma_profiles IDS in the provided IMAS DBEntry."""

    plasma = build_plasma_profiles_ids(solution, metadata, grid)
    entry.put(plasma, metadata.occurrence)
    return plasma
