import json

import numpy as np

from .config import IMASExportMetadata, RectangularGrid2D
from .common import add_constant_zeff_metadata, rectangular_grid_metadata, set_constant_zeff, store_node_field, store_parallel_velocity
from .evaluate import evaluate_variables_on_grid
from .ggd_geometry import populate_rectangular_grid_ggd_array
from hdg_postprocess.core.solution.impurity_radiation import get_impurity_radiation_metadata


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

    sampled = sample_plasma_fields(solution, grid)
    populate_plasma_ggd(ggd, sampled, grid_index=1)

    set_constant_zeff(plasma, solution, count=1)

    plasma.code.name = "SOLEDGE-HDG"
    plasma.code.repository = "hdg_postprocess"
    plasma.code.description = "Plasma profiles exported from SOLEDGE-HDG by hdg_postprocess."
    plasma.code.parameters = json.dumps(build_plasma_profiles_metadata(solution, grid), sort_keys=True)
    return plasma


def put_plasma_profiles(entry, solution, metadata: IMASExportMetadata, grid: RectangularGrid2D):
    """Build and store one plasma_profiles IDS in the provided IMAS DBEntry."""

    plasma = build_plasma_profiles_ids(solution, metadata, grid)
    entry.put(plasma, metadata.occurrence)
    return plasma


def sample_plasma_fields(solution, grid):
    r_grid, z_grid = grid.mesh()
    return evaluate_variables_on_grid(
        solution,
        r_grid,
        z_grid,
        ["n", "te", "ti", "u", "nn", "psi"],
        locator=solution.mesh.geometry.element_locator,
        outside_value=np.nan,
    )


def populate_plasma_ggd(ggd, sampled, *, grid_index):
    store_node_field(ggd.electrons.density, sampled["n"], grid_index=grid_index)
    store_node_field(ggd.electrons.temperature, sampled["te"], grid_index=grid_index)

    ggd.ion.resize(1)
    ion = ggd.ion[0]
    ion.name = "D+"
    ion.z_ion = 1.0
    store_node_field(ion.temperature, sampled["ti"], grid_index=grid_index)
    store_parallel_velocity(ion.velocity, sampled["u"], grid_index=grid_index)

    ggd.neutral.resize(1)
    neutral = ggd.neutral[0]
    neutral.name = "D"
    store_node_field(neutral.density, sampled["nn"], grid_index=grid_index)

    store_node_field(ggd.psi, sampled["psi"], grid_index=grid_index)


def build_plasma_profiles_metadata(solution, grid):
    extracted = rectangular_grid_metadata(grid)
    extracted.update(
        {
        "ggd_grid_subset": "All exported plasma fields currently live on the nodes subset.",
        "outside_mesh_policy": "Values outside the HDG mesh are exported as NaN.",
        "model_note": "SOLEDGE-HDG currently uses shared plasma density and parallel velocity for electrons and the single ion species.",
        "density_storage_note": (
            "For the current single-ion SOLEDGE-HDG model, electrons.density is the authoritative density field. "
            "The redundant ion[0].density and n_i_total fields are intentionally left empty to reduce storage."
        ),
        "temperature_storage_note": (
            "For the current single-ion model, ion[0].temperature is populated and the redundant t_i_average field is left empty."
        ),
    })
    add_constant_zeff_metadata(extracted, solution)
    impurity_metadata = get_impurity_radiation_metadata(solution, require_coefficients=False)
    if impurity_metadata is not None:
        extracted["n_impurities"] = impurity_metadata.n_impurities
        extracted["impurity_names"] = list(impurity_metadata.impurity_names)
        extracted["impurity_concentrations"] = impurity_metadata.impurity_concentrations.tolist()
        if impurity_metadata.n_impurities == 1:
            extracted["impurity_name"] = impurity_metadata.impurity_names[0]
    return extracted
