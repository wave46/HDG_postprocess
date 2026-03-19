import numpy as np


def rectangular_grid_metadata(grid):
    return {
        "grid_shape": [grid.nr, grid.nz],
        "grid_r_range_m": [grid.r_min, grid.r_max],
        "grid_z_range_m": [grid.z_min, grid.z_max],
        "ggd_grid_name": "rectangular_rz",
        "ggd_grid_subset": "All exported fields currently live on the nodes subset.",
        "value_ordering": "Node values are flattened from meshgrid(indexing='ij') in C order, so R is the slow axis and Z the fast axis.",
    }


def extend_time_metadata(metadata_dict, times):
    metadata_dict["snapshot_count"] = len(times)
    metadata_dict["exported_times_s"] = list(times)
    metadata_dict["grid_count"] = len(times)
    metadata_dict["time_varying_grid"] = True
    return metadata_dict


def add_constant_zeff_metadata(metadata_dict, solution):
    if "Zeff" in solution.parameters["physics"]:
        metadata_dict["Zeff"] = float(solution.parameters["physics"]["Zeff"])
        metadata_dict["Zeff_storage_note"] = (
            "Spatially constant Zeff is exported through plasma_profiles.global_quantities.z_eff_resistive."
        )
    return metadata_dict


def store_node_field(field_container, values, *, grid_index, flatten=True):
    field_container.resize(1)
    field_container[0].grid_index = int(grid_index)
    field_container[0].grid_subset_index = 1
    field_container[0].values = values.reshape(-1) if flatten else values


def store_parallel_velocity(velocity_container, values, *, grid_index):
    velocity_container.resize(1)
    velocity_container[0].grid_index = int(grid_index)
    velocity_container[0].grid_subset_index = 1
    velocity_container[0].parallel = values.reshape(-1)


def set_constant_zeff(plasma, solution, *, count):
    if "Zeff" in solution.parameters["physics"]:
        plasma.global_quantities.z_eff_resistive = np.full(
            int(count),
            float(solution.parameters["physics"]["Zeff"]),
            dtype=float,
        )
