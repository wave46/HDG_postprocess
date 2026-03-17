import numpy as np


def _global_mesh_sizes(mesh):
    n_elements = 0
    n_vertices = 0
    for partition_data in mesh.raw.rest_mesh_data:
        n_elements = max(n_elements, partition_data["loc2glob_el"].max())
        n_vertices = max(n_vertices, partition_data["loc2glob_no"].max())
    return n_elements + 1, n_vertices + 1


def _fill_global_mesh(mesh):
    for partition_index in range(mesh.n_partitions):
        partition_data = mesh.raw.rest_mesh_data[partition_index]
        non_ghost = ~mesh.raw.ghost_elements[partition_index].astype(bool).flatten()
        global_elements = partition_data["loc2glob_el"][non_ghost]
        local_connectivity = mesh.raw.connectivity[partition_index][non_ghost, :]
        mesh.global_state.connectivity[global_elements] = partition_data["loc2glob_no"][local_connectivity]
        mesh.global_state.vertices[partition_data["loc2glob_no"], :] = mesh.raw.vertices[partition_index]


def recombine_full_mesh(mesh):
    n_elements, n_vertices = _global_mesh_sizes(mesh)
    mesh.global_state.n_elements = n_elements
    mesh.global_state.n_vertices = n_vertices
    mesh.global_state.connectivity = np.zeros((n_elements, mesh.mesh_parameters["nodes_per_element"]), dtype=int)
    mesh.global_state.vertices = np.zeros((n_vertices, mesh.mesh_parameters["Ndim"]))
    _fill_global_mesh(mesh)
    mesh.metadata.flags.combined_to_full = True


def _single_partition_boundary_rows(mesh, raw_boundary_info):
    return (
        mesh.raw.connectivity_boundary[0],
        raw_boundary_info[0]["boundary_flags"],
        raw_boundary_info[0]["exterior_faces"][:, 0][None].T,
        raw_boundary_info[0]["exterior_faces"][:, 1],
        None,
    )


def _partitioned_boundary_rows(mesh, raw_boundary_info):
    n_faces = 0
    for partition_data in mesh.raw.rest_mesh_data:
        n_faces = max(n_faces, partition_data["loc2glob_fa"].max())
    n_faces += 1
    mesh.global_state.n_faces = n_faces

    connectivity = -1 * np.ones((n_faces, mesh.mesh_parameters["nodes_per_face"]), dtype=int)
    boundary_flags = -1 * np.ones(n_faces, dtype=int)
    face_element_number = np.zeros((n_faces, 1), dtype=int)
    face_local_number = np.zeros(n_faces, dtype=int)

    for partition_index in range(mesh.n_partitions):
        partition_data = mesh.raw.rest_mesh_data[partition_index]
        next_faces = mesh.raw.mesh_numbers[partition_index]["Nextfaces"]
        global_faces = partition_data["loc2glob_fa"][-next_faces:]
        non_ghost = (~mesh.raw.ghost_faces[partition_index].flatten())[-next_faces:]
        selected_faces = global_faces[non_ghost]

        connectivity[selected_faces, :] = partition_data["loc2glob_no"][
            mesh.raw.connectivity_boundary[partition_index][:, :]
        ][non_ghost]
        boundary_flags[selected_faces] = raw_boundary_info[partition_index]["boundary_flags"][non_ghost]
        face_element_number[selected_faces] = partition_data["loc2glob_el"][
            raw_boundary_info[partition_index]["exterior_faces"][:, 0][non_ghost]
        ][None].T
        face_local_number[selected_faces] = raw_boundary_info[partition_index]["exterior_faces"][:, 1][non_ghost]

    filled = (connectivity != -1).all(axis=1)
    return (
        connectivity[filled, :],
        boundary_flags[filled],
        face_element_number[filled],
        face_local_number[filled],
        filled,
    )


def _collect_boundary_components(bound_connectivity, boundary_idx):
    starting_indices = bound_connectivity[:, 0]
    ending_indices = bound_connectivity[:, -1]
    difference = np.setdiff1d(starting_indices, ending_indices)

    components = []
    visited = set()
    while len(visited) < len(bound_connectivity):
        if len(difference) > 0:
            starting_ind = difference[0]
            difference = difference[1:]
        else:
            unvisited_idx = next(idx for idx in range(len(bound_connectivity)) if idx not in visited)
            starting_ind = starting_indices[unvisited_idx]

        component = []
        i = np.where(starting_ind == starting_indices)[0][0]
        while i not in visited:
            component.append(boundary_idx[i])
            visited.add(i)
            next_candidates = np.where(bound_connectivity[i, -1] == bound_connectivity[:, 0])[0]
            if len(next_candidates) == 0:
                break
            i = next_candidates[0]
        components.append(component)
    return components


def _group_boundary_state(mesh, connectivity, boundary_flags, face_element_number, face_local_number):
    mesh.boundary_state.connectivity = {}
    mesh.boundary_state.flags = {}
    mesh.boundary_state.face_element_number = {}
    mesh.boundary_state.face_local_number = {}
    indices = {}

    for boundary_type in np.unique(boundary_flags):
        boundary_idx = np.where(boundary_flags == boundary_type)[0]
        bound_connectivity = connectivity[boundary_idx, :]
        components = _collect_boundary_components(bound_connectivity, boundary_idx)

        mesh.boundary_state.connectivity[boundary_type] = []
        mesh.boundary_state.flags[boundary_type] = []
        mesh.boundary_state.face_element_number[boundary_type] = []
        mesh.boundary_state.face_local_number[boundary_type] = []

        for component in components:
            mesh.boundary_state.connectivity[boundary_type].append(connectivity[component, :])
            mesh.boundary_state.flags[boundary_type].append(boundary_flags[component])
            mesh.boundary_state.face_element_number[boundary_type].append(face_element_number[component])
            mesh.boundary_state.face_local_number[boundary_type].append(face_local_number[component])
        indices[boundary_type] = components

    mesh.boundary_state.indices = indices


def _reset_boundary_gauss_cache(mesh):
    mesh.boundary_state.vertices_gauss = None
    mesh.boundary_state.tangentials_gauss = None
    mesh.boundary_state.normals_gauss = None
    mesh.boundary_state.segment_length_gauss = None
    mesh.boundary_state.segment_surface_gauss = None
    mesh.metadata.flags.boundary_gauss_initialized = False
    mesh.metadata.cache.boundary_gauss_boundaries = None


def recombine_full_boundary(mesh, raw_boundary_info):
    if mesh.n_partitions > 1:
        connectivity, boundary_flags, face_element_number, face_local_number, filled = _partitioned_boundary_rows(mesh, raw_boundary_info)
        mesh.boundary_state.filled = filled
    else:
        connectivity, boundary_flags, face_element_number, face_local_number, _ = _single_partition_boundary_rows(mesh, raw_boundary_info)
        mesh.global_state.n_faces = connectivity.shape[0]

    _group_boundary_state(mesh, connectivity, boundary_flags, face_element_number, face_local_number)
    _reset_boundary_gauss_cache(mesh)
    mesh.metadata.flags.boundary_combined = True


def calculate_gauss_boundary(mesh, boundaries, raw_boundary_info):
    requested_boundaries = tuple(boundaries)
    if (
        mesh.metadata.flags.boundary_gauss_initialized
        and mesh.metadata.cache.boundary_gauss_boundaries == requested_boundaries
    ):
        from hdg_postprocess.core.mesh.boundary import boundary_ordering

        return boundary_ordering(mesh, raw_boundary_info, boundaries)

    if mesh.metadata.reference_element is None:
        raise ValueError("Please, provide reference element")
    if not mesh.metadata.flags.combined_to_full:
        recombine_full_mesh(mesh)
    if not mesh.metadata.flags.boundary_combined:
        if raw_boundary_info is None:
            raise ValueError("Please, provide raw boundary info as input to this method")
        recombine_full_boundary(mesh, raw_boundary_info)

    from hdg_postprocess.core.mesh.boundary import boundary_ordering

    boundary_ordering_res, connectivity_ordered, iel_face_number = boundary_ordering(mesh, raw_boundary_info, boundaries)

    mesh.boundary_state.vertices_gauss = np.einsum(
        "ij,kjh->kih", mesh.metadata.reference_element["N1d"], mesh.global_state.vertices[connectivity_ordered, :]
    )
    derivative_gauss = np.einsum(
        "ij,kjh->kih", mesh.metadata.reference_element["N1dxi"], mesh.global_state.vertices[connectivity_ordered, :]
    )
    derivative_norm = np.sqrt(((derivative_gauss ** 2).sum(axis=2)))[:, :, None]
    mesh.boundary_state.tangentials_gauss = derivative_gauss / derivative_norm
    mesh.boundary_state.normals_gauss = np.zeros_like(mesh.boundary_state.tangentials_gauss)
    mesh.boundary_state.normals_gauss[:, :, 0] = mesh.boundary_state.tangentials_gauss[:, :, 1]
    mesh.boundary_state.normals_gauss[:, :, 1] = -1 * mesh.boundary_state.tangentials_gauss[:, :, 0]
    mesh.boundary_state.segment_length_gauss = derivative_norm * mesh.metadata.reference_element["IPweights1d"][None, :, :]
    mesh.boundary_state.segment_surface_gauss = (
        mesh.boundary_state.segment_length_gauss * 2 * np.pi * mesh.boundary_state.vertices_gauss[:, :, 0][:, :, None]
    )
    mesh.metadata.flags.boundary_gauss_initialized = True
    mesh.metadata.cache.boundary_gauss_boundaries = requested_boundaries
    return boundary_ordering_res, connectivity_ordered, iel_face_number
