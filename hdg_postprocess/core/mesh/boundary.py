import numpy as np


def _ensure_boundary_combined(mesh, raw_boundary_info):
    if mesh.metadata.flags.boundary_combined:
        return
    if raw_boundary_info is None:
        raise ValueError("Please, provide raw boundary info as input to this method")
    from hdg_postprocess.core.mesh import assembly as assembly_ops

    assembly_ops.recombine_full_boundary(mesh, raw_boundary_info)


def boundary_ordering(mesh, raw_boundary_info, boundaries):
    _ensure_boundary_combined(mesh, raw_boundary_info)

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
