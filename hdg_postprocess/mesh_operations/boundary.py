import numpy as np


def recombine_full_boundary(mesh, raw_boundary_info):
    if mesh.n_partitions > 1:
        mesh.global_state.n_faces = 0
        for i in range(mesh.n_partitions):
            mesh.global_state.n_faces = max(mesh.global_state.n_faces, mesh.raw_rest_mesh_data[i]["loc2glob_fa"].max())
        mesh.global_state.n_faces += 1

        connectivity_b_glob = -1 * np.ones((mesh.global_state.n_faces, mesh.mesh_parameters["nodes_per_face"]), dtype=int)
        boundary_flags = -1 * np.ones((mesh.global_state.n_faces), dtype=int)
        face_element_number = np.zeros((mesh.global_state.n_faces, 1), dtype=int)
        face_local_number = np.zeros((mesh.global_state.n_faces), dtype=int)

        for i in range(mesh.n_partitions):
            non_ghost = (~mesh.raw_ghost_faces[i].flatten())[-mesh.raw_mesh_numbers[i]["Nextfaces"]:]

            connectivity_b_glob[mesh.raw_rest_mesh_data[i]["loc2glob_fa"][-mesh.raw_mesh_numbers[i]["Nextfaces"]:][non_ghost], :] = (
                mesh.raw_rest_mesh_data[i]["loc2glob_no"][mesh.raw_connectivity_boundary[i][:, :]][non_ghost]
            )
            boundary_flags[mesh.raw_rest_mesh_data[i]["loc2glob_fa"][-mesh.raw_mesh_numbers[i]["Nextfaces"]:][non_ghost]] = (
                raw_boundary_info[i]["boundary_flags"][non_ghost]
            )
            face_element_number[mesh.raw_rest_mesh_data[i]["loc2glob_fa"][-mesh.raw_mesh_numbers[i]["Nextfaces"]:][non_ghost]] = (
                mesh.raw_rest_mesh_data[i]["loc2glob_el"][raw_boundary_info[i]["exterior_faces"][:, 0][non_ghost]][None].T
            )
            face_local_number[mesh.raw_rest_mesh_data[i]["loc2glob_fa"][-mesh.raw_mesh_numbers[i]["Nextfaces"]:][non_ghost]] = (
                raw_boundary_info[i]["exterior_faces"][:, 1][non_ghost]
            )

        filled = (connectivity_b_glob != -1).all(axis=1)
        connectivity_b_glob = connectivity_b_glob[filled, :]
        boundary_flags = boundary_flags[filled]
        face_element_number = face_element_number[filled]
        face_local_number = face_local_number[filled]
        mesh.boundary_state.filled = filled
    else:
        mesh.global_state.n_faces = mesh._raw_connectivity_boundary[0].shape[0]
        connectivity_b_glob = mesh._raw_connectivity_boundary[0]
        boundary_flags = raw_boundary_info[0]["boundary_flags"]
        face_element_number = raw_boundary_info[0]["exterior_faces"][:, 0][None].T
        face_local_number = raw_boundary_info[0]["exterior_faces"][:, 1]

    unique_boundaries = np.unique(boundary_flags)
    indices = {}
    mesh.boundary_state.connectivity = {}
    mesh.boundary_state.flags = {}
    mesh.boundary_state.face_element_number = {}
    mesh.boundary_state.face_local_number = {}

    for boundary_type in unique_boundaries:
        boundary_idx = np.where(boundary_flags == boundary_type)[0]
        bound_connectivity = connectivity_b_glob[boundary_idx, :]

        starting_indices = bound_connectivity[:, 0]
        ending_indices = bound_connectivity[:, -1]
        difference = np.setdiff1d(starting_indices, ending_indices)

        all_ind = []
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

            all_ind.append(component)

        mesh.boundary_state.connectivity[boundary_type] = []
        mesh.boundary_state.flags[boundary_type] = []
        mesh.boundary_state.face_element_number[boundary_type] = []
        mesh.boundary_state.face_local_number[boundary_type] = []

        for component in all_ind:
            mesh.boundary_state.connectivity[boundary_type].append(connectivity_b_glob[component, :])
            mesh.boundary_state.flags[boundary_type].append(boundary_flags[component])
            mesh.boundary_state.face_element_number[boundary_type].append(face_element_number[component])
            mesh.boundary_state.face_local_number[boundary_type].append(face_local_number[component])

        indices[boundary_type] = all_ind

    mesh.boundary_state.indices = indices
    mesh.boundary_state.vertices_gauss = None
    mesh.boundary_state.tangentials_gauss = None
    mesh.boundary_state.normals_gauss = None
    mesh.boundary_state.segment_length_gauss = None
    mesh.boundary_state.segment_surface_gauss = None
    mesh.metadata.flags.boundary_combined = True
    mesh.metadata.flags.boundary_gauss_initialized = False
    mesh.metadata.cache.boundary_gauss_boundaries = None


def boundary_ordering(mesh, raw_boundary_info, boundaries):
    if not mesh.metadata.flags.boundary_combined:
        recombine_full_boundary(mesh, raw_boundary_info)

    connected_boundaries = []
    segments = 0
    for boundary in boundaries:
        connected_boundary = []
        for _sub_connectivity in mesh.boundary_state.connectivity[boundary]:
            connected_boundary.append(False)
            segments += 1
        connected_boundaries.append(connected_boundary)

    boundary_ordering_res = [[0, 0]]
    connected_boundaries[0][0] = True
    segments -= 1
    node_idx = mesh.boundary_state.connectivity[boundaries[0]][0][-1, -1]

    looped_segments = []
    while segments > 0:
        found_segment = False
        for i, connected_boundary in enumerate(connected_boundaries):
            for j, segment in enumerate(connected_boundary):
                if not segment and node_idx == mesh.boundary_state.connectivity[boundaries[i]][j][0, 0]:
                    connected_boundaries[i][j] = True
                    segments -= 1
                    boundary_ordering_res.append([i, j])
                    node_idx = mesh.boundary_state.connectivity[boundaries[i]][j][-1, -1]
                    found_segment = True
                    break
            if found_segment:
                break

        if not found_segment:
            for i, connected_boundary in enumerate(connected_boundaries):
                for j, segment in enumerate(connected_boundary):
                    if not segment:
                        connected_boundaries[i][j] = True
                        segments -= 1
                        looped_segments.append([i, j])
                        node_idx = mesh.boundary_state.connectivity[boundaries[i]][j][-1, -1]
                        break
                if found_segment:
                    break

    boundary_ordering_res.extend(looped_segments)

    connectivity_b_ordered = np.empty([0, mesh.boundary_state.connectivity[boundaries[0]][0].shape[1]], dtype=int)
    iel_face_ordered = np.empty([0, mesh.boundary_state.face_element_number[boundaries[0]][0].shape[1]], dtype=int)
    for bound_order in boundary_ordering_res:
        connectivity_b_ordered = np.vstack(
            [connectivity_b_ordered, mesh.boundary_state.connectivity[boundaries[bound_order[0]]][bound_order[1]]]
        )
        iel_face_ordered = np.vstack(
            [iel_face_ordered, mesh.boundary_state.face_element_number[boundaries[bound_order[0]]][bound_order[1]]]
        )

    return boundary_ordering_res, connectivity_b_ordered, iel_face_ordered


def calculate_gauss_boundary(mesh, boundaries, raw_boundary_info):
    requested_boundaries = tuple(boundaries)
    if (
        mesh.metadata.flags.boundary_gauss_initialized
        and mesh.metadata.cache.boundary_gauss_boundaries == requested_boundaries
    ):
        boundary_ordering_res, connectivity_ordered, iel_face_number = boundary_ordering(mesh, raw_boundary_info, boundaries)
        return boundary_ordering_res, connectivity_ordered, iel_face_number

    if mesh.metadata.reference_element is None:
        raise ValueError("Please, provide reference element")
    if not mesh.metadata.flags.combined_to_full:
        from hdg_postprocess.mesh_operations.geometry import recombine_full_mesh

        recombine_full_mesh(mesh)
    if not mesh.metadata.flags.boundary_combined:
        if raw_boundary_info is None:
            raise ValueError("Please, provide raw boundary info as input to this method")
        recombine_full_boundary(mesh, raw_boundary_info)
    boundary_ordering_res, connectivity_ordered, iel_face_number = boundary_ordering(mesh, raw_boundary_info, boundaries)

    mesh.boundary_state.vertices_gauss = np.einsum(
        "ij,kjh->kih", mesh.metadata.reference_element["N1d"], mesh.vertices_glob[connectivity_ordered, :]
    )
    derivative_gauss = np.einsum(
        "ij,kjh->kih", mesh.metadata.reference_element["N1dxi"], mesh.vertices_glob[connectivity_ordered, :]
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
