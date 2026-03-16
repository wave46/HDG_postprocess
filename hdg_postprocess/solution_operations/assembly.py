import numpy as np


def recombine_full_solution(solution):
    if not solution.mesh.combined_to_full:
        print("Comibining first mesh full")
        solution.mesh.recombine_full_mesh()

    solution._solution_glob = np.zeros(
        (solution.mesh._nelems_glob, solution.mesh.mesh_parameters["nodes_per_element"], solution.neq)
    )
    solution._gradient_glob = np.zeros(
        (solution.mesh._nelems_glob, solution.mesh.mesh_parameters["nodes_per_element"], solution.neq, solution.ndim)
    )
    solution._magnetic_field_glob = np.zeros(
        (solution.mesh._nelems_glob, solution.mesh.mesh_parameters["nodes_per_element"], 3)
    )
    if "poloidal_flux" in solution.raw_equilibriums[0].keys():
        solution._poloidal_flux_glob = np.zeros(
            (solution.mesh._nelems_glob, solution.mesh.mesh_parameters["nodes_per_element"])
        )
    if solution.parameters["switches"]["ohmicsrc"][0] == 1:
        solution._jtor_glob = np.zeros(
            (solution.mesh._nelems_glob, solution.mesh.mesh_parameters["nodes_per_element"])
        )

    for i in range(solution.n_partitions):
        raw_solution = solution.raw_solutions[i].reshape(
            solution.raw_solutions[i].shape[0] // solution.mesh.mesh_parameters["nodes_per_element"],
            solution.mesh.mesh_parameters["nodes_per_element"],
            solution.neq,
        )
        raw_gradient = solution.raw_gradients[i].reshape(
            solution.raw_gradients[i].shape[0] // solution.mesh.mesh_parameters["nodes_per_element"],
            solution.mesh.mesh_parameters["nodes_per_element"],
            solution.neq,
            solution.ndim,
        )
        raw_field = solution.raw_equilibriums[i]["magnetic_field"][solution.mesh.raw_connectivity[i]]

        if "poloidal_flux" in solution.raw_equilibriums[0].keys():
            raw_poloidal_flux = solution.raw_equilibriums[i]["poloidal_flux"][solution.mesh.raw_connectivity[i]]

        if solution.parameters["switches"]["ohmicsrc"][0] == 1:
            raw_jtor = solution.raw_equilibriums[i]["plasma_current"][solution.mesh.raw_connectivity[i]]

        if solution.n_partitions > 1:
            mask = ~solution.mesh.raw_ghost_elements[i].astype(bool).flatten()
            solution._solution_glob[solution.mesh.raw_rest_mesh_data[i]["loc2glob_el"][mask]] = raw_solution[mask, :]
            solution._gradient_glob[solution.mesh.raw_rest_mesh_data[i]["loc2glob_el"][mask]] = raw_gradient[mask, :, :]
            solution._magnetic_field_glob[solution.mesh.raw_rest_mesh_data[i]["loc2glob_el"][mask]] = raw_field[mask, :]
            if "poloidal_flux" in solution.raw_equilibriums[0].keys():
                solution._poloidal_flux_glob[solution.mesh.raw_rest_mesh_data[i]["loc2glob_el"][mask]] = raw_poloidal_flux[mask]
            if solution.parameters["switches"]["ohmicsrc"][0] == 1:
                solution._jtor_glob[solution.mesh.raw_rest_mesh_data[i]["loc2glob_el"][mask]] = raw_jtor[mask, :]
        else:
            solution._solution_glob = raw_solution
            solution._gradient_glob = raw_gradient
            solution._magnetic_field_glob = raw_field
            if "poloidal_flux" in solution.raw_equilibriums[0].keys():
                solution._poloidal_flux_glob = raw_poloidal_flux
            if solution.parameters["switches"]["ohmicsrc"][0] == 1:
                solution._jtor_glob = raw_jtor

    if "external_heating" in solution.parameters["physics"]:
        solution._external_heating = solution.parameters["physics"]["external_heating"][solution.mesh._connectivity_glob]
    if "external_heating_e" in solution.parameters["physics"]:
        solution._external_heating_e = solution.parameters["physics"]["external_heating_e"][solution.mesh._connectivity_glob]
    if "external_heating_i" in solution.parameters["physics"]:
        solution._external_heating_i = solution.parameters["physics"]["external_heating_i"][solution.mesh._connectivity_glob]

    solution._magnetic_field_unit_glob = solution._magnetic_field_glob / np.sqrt(
        (solution._magnetic_field_glob ** 2).sum(axis=-1)
    )[:, :, None]
    solution._combined_to_full = True


def recombine_simple_full_solution(solution):
    if not solution.combined_to_full:
        print("Comibining first solution full")
        solution.recombine_full_solution()
    solution._solution_simple = np.zeros([solution.mesh.vertices_glob.shape[0], solution.neq])
    solution._solution_simple[solution.mesh.connectivity_glob.reshape(-1, 1).ravel(), :] = solution.solution_glob.reshape(
        solution.solution_glob.shape[0] * solution.solution_glob.shape[1], solution.neq
    )

    solution._gradient_simple = np.zeros([solution.mesh.vertices_glob.shape[0], solution.neq, solution.ndim])
    solution._gradient_simple[solution.mesh.connectivity_glob.reshape(-1, 1).ravel(), :, :] = solution.gradient_glob.reshape(
        solution.gradient_glob.shape[0] * solution.gradient_glob.shape[1], solution.neq, solution.ndim
    )

    solution._magnetic_field_simple = np.zeros([solution.mesh.vertices_glob.shape[0], 3])
    solution._magnetic_field_simple[solution.mesh.connectivity_glob.reshape(-1, 1).ravel(), :] = solution.magnetic_field_glob.reshape(
        solution.magnetic_field_glob.shape[0] * solution.magnetic_field_glob.shape[1], 3
    )
    if solution.parameters["switches"]["ohmicsrc"][0] == 1:
        solution._jtor_simple = np.zeros(solution.mesh.vertices_glob.shape[0])
        solution._jtor_simple[solution.mesh.connectivity_glob.reshape(-1, 1).ravel()] = solution.jtor_glob.reshape(
            solution.jtor_glob.shape[0] * solution.jtor_glob.shape[1]
        )
    if "poloidal_flux" in solution.raw_equilibriums[0].keys():
        solution._poloidal_flux_simple = np.zeros([solution.mesh.vertices_glob.shape[0]])
        solution._poloidal_flux_simple[solution.mesh.connectivity_glob.reshape(-1, 1).ravel()] = solution.poloidal_flux_glob.reshape(
            solution.poloidal_flux_glob.shape[0] * solution.poloidal_flux_glob.shape[1]
        )
    solution._combined_simple_solution = True

    if "external_heating" in solution.parameters["physics"]:
        solution._external_heating_simple = solution.parameters["physics"]["external_heating"]
    if "external_heating_e" in solution.parameters["physics"]:
        solution._external_heating_e_simple = solution.parameters["physics"]["external_heating_e"]
    if "external_heating_i" in solution.parameters["physics"]:
        solution._external_heating_i_simple = solution.parameters["physics"]["external_heating_i"]


def recombine_boundary_solution(solution):
    if not solution.combined_to_full:
        print("Comibining first solution full")
        solution.recombine_full_solution()
    if solution.mesh.connectivity_b_glob is None:
        print("Comibining first boundary connectivity and info")
        solution.mesh.recombine_full_boundary(solution.raw_solution_boundary_infos)
    if solution.mesh.reference_element is None:
        raise ValueError("Please, provide reference element to the mesh")

    solution._solution_boundary = {}
    solution._gradient_boundary = {}
    solution._magnetic_field_boundary = {}
    solution._magnetic_field_unit_boundary = {}
    solution._poloidal_flux_boundary = {}
    solution._solution_skeleton_boundary = {}

    for key in solution.mesh._connectivity_b_glob.keys():
        solution._solution_boundary[key] = []
        solution._gradient_boundary[key] = []
        solution._magnetic_field_boundary[key] = []
        solution._poloidal_flux_boundary[key] = []
        solution._magnetic_field_unit_boundary[key] = []
        for face_element_number, face_local_number in zip(solution.mesh.face_element_number[key], solution.mesh.face_local_number[key]):
            face_nodes = solution.mesh.reference_element["faceNodes"][face_local_number, :]
            solution._solution_boundary[key].append(solution.solution_glob[face_element_number, face_nodes, :])
            solution._gradient_boundary[key].append(solution.gradient_glob[face_element_number, face_nodes, :, :])
            solution._magnetic_field_boundary[key].append(solution.magnetic_field_glob[face_element_number, face_nodes, :])
            solution._magnetic_field_unit_boundary[key].append(solution.magnetic_field_unit_glob[face_element_number, face_nodes, :])
            solution._poloidal_flux_boundary[key].append(solution.poloidal_flux_glob[face_element_number, face_nodes])

    if solution.n_partitions > 1:
        solution_skeleton_boundary = np.ones((solution.mesh._nfaces_glob, solution.mesh.mesh_parameters["nodes_per_face"], solution.neq))
        for i in range(solution.n_partitions):
            non_ghost = ~solution.mesh.raw_ghost_faces[i].flatten()
            raw_solution = solution.raw_solutions_skeleton[i].reshape(
                solution.raw_solutions_skeleton[i].shape[0] // solution.mesh.mesh_parameters["nodes_per_face"],
                solution.mesh.mesh_parameters["nodes_per_face"],
                solution.neq,
            )
            solution_skeleton_boundary[solution.mesh.raw_rest_mesh_data[i]["loc2glob_fa"][:][non_ghost], :] = raw_solution[non_ghost]
        solution_skeleton_boundary = solution_skeleton_boundary[solution.mesh._filled, :, :]
    else:
        solution_skeleton_boundary = solution.raw_solutions_skeleton[0].reshape(
            solution.raw_solutions_skeleton[0].shape[0] // solution.mesh.mesh_parameters["nodes_per_face"],
            solution.mesh.mesh_parameters["nodes_per_face"],
            solution.neq,
        )
        solution_skeleton_boundary = solution_skeleton_boundary[-solution.mesh.raw_mesh_numbers[0]["Nextfaces"] :, :, :]
    solution._solution_skeleton_boundary = {}
    print(solution_skeleton_boundary.shape)
    for key, indices in solution.mesh._indices.items():
        solution._solution_skeleton_boundary[key] = []
        for ind in indices:
            solution._solution_skeleton_boundary[key].append(solution_skeleton_boundary[ind, :, :])

    solution._combined_boundary = True


def calculate_in_gauss_points(solution):
    if not solution.combined_to_full:
        print("Comibining first solution full")
        solution.recombine_full_solution()
    if solution.mesh.reference_element is None:
        raise ValueError("Please, provide reference element to the mesh")

    solution._solution_gauss = np.einsum("ij,kjh->kih", solution.mesh.reference_element["N"], solution.solution_glob)
    solution._gradient_gauss = np.einsum("ij,kjhl->kihl", solution.mesh.reference_element["N"], solution.gradient_glob)
    solution._magnetic_field_gauss = np.einsum("ij,kjh->kih", solution.mesh.reference_element["N"], solution.magnetic_field_glob)
    solution._magnetic_field_unit_gauss = np.einsum(
        "ij,kjh->kih", solution.mesh.reference_element["N"], solution.magnetic_field_unit_glob
    )
    solution._jtor_gauss = np.einsum("ij,kj->ki", solution.mesh.reference_element["N"], solution.jtor_glob)
    solution._poloidal_flux_gauss = np.einsum("ij,kj->ki", solution.mesh.reference_element["N"], solution.poloidal_flux_glob)

    if "external_heating" in solution.parameters["physics"]:
        solution._external_heating_gauss = np.einsum("ij,kj->ki", solution.mesh.reference_element["N"], solution.external_heating)
    if "external_heating_e" in solution.parameters["physics"]:
        solution._external_heating_e_gauss = np.einsum("ij,kj->ki", solution.mesh.reference_element["N"], solution.external_heating_e)
    if "external_heating_i" in solution.parameters["physics"]:
        solution._external_heating_i_gauss = np.einsum("ij,kj->ki", solution.mesh.reference_element["N"], solution.external_heating_i)


def calculate_in_boundary_gauss_points(solution, boundaries):
    if solution.mesh.reference_element is None:
        raise ValueError("Please, provide reference element to the mesh")
    if not solution.combined_boundary:
        print("Comibining first values on boundary")
        solution.recombine_boundary_solution()
    boundary_ordering, connectivity_ordered, iel_face_ordered = solution.mesh.calculate_gauss_boundary(
        boundaries, solution.raw_solution_boundary_infos
    )
    boundary_view = solution.views.boundary
    boundary_solution = boundary_view.solution.conservative
    boundary_solution_skeleton = boundary_view.solution_skeleton.conservative
    boundary_gradient = boundary_view.gradient.conservative
    boundary_equilibrium = boundary_view.equilibrium

    solution_boundary_ordered = np.empty(
        [0, boundary_solution[boundaries[0]][0].shape[1], boundary_solution[boundaries[0]][0].shape[2]]
    )
    solution_skeleton_boundary_ordered = np.empty(
        [0, boundary_solution_skeleton[boundaries[0]][0].shape[1], boundary_solution_skeleton[boundaries[0]][0].shape[2]]
    )
    gradient_boundary_ordered = np.empty(
        [0, boundary_gradient[boundaries[0]][0].shape[1], boundary_gradient[boundaries[0]][0].shape[2], boundary_gradient[boundaries[0]][0].shape[3]]
    )
    magnetic_field_boundary_ordered = np.empty(
        [0, boundary_equilibrium.magnetic_field[boundaries[0]][0].shape[1], boundary_equilibrium.magnetic_field[boundaries[0]][0].shape[2]]
    )
    magnetic_field_unit_boundary_ordered = np.empty(
        [0, boundary_equilibrium.magnetic_field_unit[boundaries[0]][0].shape[1], boundary_equilibrium.magnetic_field_unit[boundaries[0]][0].shape[2]]
    )
    poloidal_flux_boundary_ordered = np.empty([0, boundary_equilibrium.poloidal_flux[boundaries[0]][0].shape[1]])
    for bound_order in boundary_ordering:
        solution_boundary_ordered = np.vstack([solution_boundary_ordered, boundary_solution[boundaries[bound_order[0]]][bound_order[1]]])
        solution_skeleton_boundary_ordered = np.vstack([
            solution_skeleton_boundary_ordered,
            boundary_solution_skeleton[boundaries[bound_order[0]]][bound_order[1]],
        ])
        gradient_boundary_ordered = np.vstack([gradient_boundary_ordered, boundary_gradient[boundaries[bound_order[0]]][bound_order[1]]])
        magnetic_field_boundary_ordered = np.vstack([magnetic_field_boundary_ordered, boundary_equilibrium.magnetic_field[boundaries[bound_order[0]]][bound_order[1]]])
        magnetic_field_unit_boundary_ordered = np.vstack([
            magnetic_field_unit_boundary_ordered,
            boundary_equilibrium.magnetic_field_unit[boundaries[bound_order[0]]][bound_order[1]],
        ])
        poloidal_flux_boundary_ordered = np.vstack([poloidal_flux_boundary_ordered, boundary_equilibrium.poloidal_flux[boundaries[bound_order[0]]][bound_order[1]]])
    solution._solution_boundary_gauss = np.einsum("ij,kjh->kih", solution.mesh.reference_element["N1d"], solution_boundary_ordered)
    solution._solution_skeleton_boundary_gauss = np.einsum("ij,kjh->kih", solution.mesh.reference_element["N1d"], solution_skeleton_boundary_ordered)
    solution._gradient_boundary_gauss = np.einsum("ij,kjhl->kihl", solution.mesh.reference_element["N1d"], gradient_boundary_ordered)
    solution._magnetic_field_boundary_gauss = np.einsum("ij,kjh->kih", solution.mesh.reference_element["N1d"], magnetic_field_boundary_ordered)
    solution._poloidal_flux_boundary_gauss = np.einsum("ij,kj->ki", solution.mesh.reference_element["N1d"], poloidal_flux_boundary_ordered)
    solution._magnetic_field_unit_boundary_gauss = np.einsum(
        "ij,kjh->kih", solution.mesh.reference_element["N1d"], magnetic_field_unit_boundary_ordered
    )
    return boundary_ordering, connectivity_ordered, iel_face_ordered
