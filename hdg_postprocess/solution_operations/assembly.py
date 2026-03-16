import numpy as np


def recombine_full_solution(solution):
    if not solution.mesh.metadata.flags.combined_to_full:
        print("Comibining first mesh full")
        solution.mesh.geometry.recombine_full()

    glob_view = solution.views.glob
    glob_view.solution.conservative = np.zeros(
        (solution.mesh.nelems_glob, solution.mesh.mesh_parameters["nodes_per_element"], solution.neq)
    )
    glob_view.gradient.conservative = np.zeros(
        (solution.mesh.nelems_glob, solution.mesh.mesh_parameters["nodes_per_element"], solution.neq, solution.ndim)
    )
    glob_view.equilibrium.magnetic_field = np.zeros(
        (solution.mesh.nelems_glob, solution.mesh.mesh_parameters["nodes_per_element"], 3)
    )
    if "poloidal_flux" in solution.raw.equilibriums[0].keys():
        glob_view.equilibrium.poloidal_flux = np.zeros(
            (solution.mesh.nelems_glob, solution.mesh.mesh_parameters["nodes_per_element"])
        )
    if solution.parameters["switches"]["ohmicsrc"][0] == 1:
        glob_view.equilibrium.jtor = np.zeros(
            (solution.mesh.nelems_glob, solution.mesh.mesh_parameters["nodes_per_element"])
        )

    for i in range(solution.n_partitions):
        raw_solution = solution.raw.solutions[i].reshape(
            solution.raw.solutions[i].shape[0] // solution.mesh.mesh_parameters["nodes_per_element"],
            solution.mesh.mesh_parameters["nodes_per_element"],
            solution.neq,
        )
        raw_gradient = solution.raw.gradients[i].reshape(
            solution.raw.gradients[i].shape[0] // solution.mesh.mesh_parameters["nodes_per_element"],
            solution.mesh.mesh_parameters["nodes_per_element"],
            solution.neq,
            solution.ndim,
        )
        raw_field = solution.raw.equilibriums[i]["magnetic_field"][solution.mesh.raw_connectivity[i]]

        if "poloidal_flux" in solution.raw.equilibriums[0].keys():
            raw_poloidal_flux = solution.raw.equilibriums[i]["poloidal_flux"][solution.mesh.raw_connectivity[i]]

        if solution.parameters["switches"]["ohmicsrc"][0] == 1:
            raw_jtor = solution.raw.equilibriums[i]["plasma_current"][solution.mesh.raw_connectivity[i]]

        if solution.n_partitions > 1:
            mask = ~solution.mesh.raw_ghost_elements[i].astype(bool).flatten()
            glob_view.solution.conservative[solution.mesh.raw_rest_mesh_data[i]["loc2glob_el"][mask]] = raw_solution[mask, :]
            glob_view.gradient.conservative[solution.mesh.raw_rest_mesh_data[i]["loc2glob_el"][mask]] = raw_gradient[mask, :, :]
            glob_view.equilibrium.magnetic_field[solution.mesh.raw_rest_mesh_data[i]["loc2glob_el"][mask]] = raw_field[mask, :]
            if "poloidal_flux" in solution.raw.equilibriums[0].keys():
                glob_view.equilibrium.poloidal_flux[solution.mesh.raw_rest_mesh_data[i]["loc2glob_el"][mask]] = raw_poloidal_flux[mask]
            if solution.parameters["switches"]["ohmicsrc"][0] == 1:
                glob_view.equilibrium.jtor[solution.mesh.raw_rest_mesh_data[i]["loc2glob_el"][mask]] = raw_jtor[mask, :]
        else:
            glob_view.solution.conservative = raw_solution
            glob_view.gradient.conservative = raw_gradient
            glob_view.equilibrium.magnetic_field = raw_field
            if "poloidal_flux" in solution.raw.equilibriums[0].keys():
                glob_view.equilibrium.poloidal_flux = raw_poloidal_flux
            if solution.parameters["switches"]["ohmicsrc"][0] == 1:
                glob_view.equilibrium.jtor = raw_jtor

    if "external_heating" in solution.parameters["physics"]:
        glob_view.sources.external_heating = solution.parameters["physics"]["external_heating"][solution.mesh.connectivity_glob]
    if "external_heating_e" in solution.parameters["physics"]:
        glob_view.sources.external_heating_e = solution.parameters["physics"]["external_heating_e"][solution.mesh.connectivity_glob]
    if "external_heating_i" in solution.parameters["physics"]:
        glob_view.sources.external_heating_i = solution.parameters["physics"]["external_heating_i"][solution.mesh.connectivity_glob]

    glob_view.equilibrium.magnetic_field_unit = glob_view.equilibrium.magnetic_field / np.sqrt(
        (glob_view.equilibrium.magnetic_field ** 2).sum(axis=-1)
    )[:, :, None]
    solution.metadata.flags.combined_to_full = True
    solution.metadata.flags.combined_gauss = False
    solution.metadata.flags.gauss_phys_initialized = False


def recombine_simple_full_solution(solution):
    if not solution.metadata.flags.combined_to_full:
        print("Comibining first solution full")
        recombine_full_solution(solution)
    glob_view = solution.views.glob
    simple_view = solution.views.simple
    simple_view.solution.conservative = np.zeros([solution.mesh.vertices_glob.shape[0], solution.neq])
    simple_view.solution.conservative[solution.mesh.connectivity_glob.reshape(-1, 1).ravel(), :] = glob_view.solution.conservative.reshape(
        glob_view.solution.conservative.shape[0] * glob_view.solution.conservative.shape[1], solution.neq
    )

    simple_view.gradient.conservative = np.zeros([solution.mesh.vertices_glob.shape[0], solution.neq, solution.ndim])
    simple_view.gradient.conservative[solution.mesh.connectivity_glob.reshape(-1, 1).ravel(), :, :] = glob_view.gradient.conservative.reshape(
        glob_view.gradient.conservative.shape[0] * glob_view.gradient.conservative.shape[1], solution.neq, solution.ndim
    )

    simple_view.equilibrium.magnetic_field = np.zeros([solution.mesh.vertices_glob.shape[0], 3])
    simple_view.equilibrium.magnetic_field[solution.mesh.connectivity_glob.reshape(-1, 1).ravel(), :] = glob_view.equilibrium.magnetic_field.reshape(
        glob_view.equilibrium.magnetic_field.shape[0] * glob_view.equilibrium.magnetic_field.shape[1], 3
    )
    if solution.parameters["switches"]["ohmicsrc"][0] == 1:
        simple_view.equilibrium.jtor = np.zeros(solution.mesh.vertices_glob.shape[0])
        simple_view.equilibrium.jtor[solution.mesh.connectivity_glob.reshape(-1, 1).ravel()] = glob_view.equilibrium.jtor.reshape(
            glob_view.equilibrium.jtor.shape[0] * glob_view.equilibrium.jtor.shape[1]
        )
    if "poloidal_flux" in solution.raw.equilibriums[0].keys():
        simple_view.equilibrium.poloidal_flux = np.zeros([solution.mesh.vertices_glob.shape[0]])
        simple_view.equilibrium.poloidal_flux[solution.mesh.connectivity_glob.reshape(-1, 1).ravel()] = glob_view.equilibrium.poloidal_flux.reshape(
            glob_view.equilibrium.poloidal_flux.shape[0] * glob_view.equilibrium.poloidal_flux.shape[1]
        )
    solution.metadata.flags.combined_simple_solution = True

    if "external_heating" in solution.parameters["physics"]:
        simple_view.sources.external_heating = solution.parameters["physics"]["external_heating"]
    if "external_heating_e" in solution.parameters["physics"]:
        simple_view.sources.external_heating_e = solution.parameters["physics"]["external_heating_e"]
    if "external_heating_i" in solution.parameters["physics"]:
        simple_view.sources.external_heating_i = solution.parameters["physics"]["external_heating_i"]


def recombine_boundary_solution(solution):
    if not solution.metadata.flags.combined_to_full:
        print("Comibining first solution full")
        recombine_full_solution(solution)
    if solution.mesh.boundary_state.connectivity is None:
        print("Comibining first boundary connectivity and info")
        solution.mesh.boundary.recombine_full(solution.raw.boundary_infos)
    if solution.mesh.metadata.reference_element is None:
        raise ValueError("Please, provide reference element to the mesh")

    boundary_view = solution.views.boundary
    boundary_view.solution.conservative = {}
    boundary_view.gradient.conservative = {}
    boundary_view.equilibrium.magnetic_field = {}
    boundary_view.equilibrium.magnetic_field_unit = {}
    boundary_view.equilibrium.poloidal_flux = {}
    boundary_view.solution_skeleton.conservative = {}
    glob_view = solution.views.glob

    for key in solution.mesh.boundary_state.connectivity.keys():
        boundary_view.solution.conservative[key] = []
        boundary_view.gradient.conservative[key] = []
        boundary_view.equilibrium.magnetic_field[key] = []
        boundary_view.equilibrium.poloidal_flux[key] = []
        boundary_view.equilibrium.magnetic_field_unit[key] = []
        for face_element_number, face_local_number in zip(
            solution.mesh.boundary_state.face_element_number[key],
            solution.mesh.boundary_state.face_local_number[key],
        ):
            face_nodes = solution.mesh.metadata.reference_element["faceNodes"][face_local_number, :]
            boundary_view.solution.conservative[key].append(glob_view.solution.conservative[face_element_number, face_nodes, :])
            boundary_view.gradient.conservative[key].append(glob_view.gradient.conservative[face_element_number, face_nodes, :, :])
            boundary_view.equilibrium.magnetic_field[key].append(glob_view.equilibrium.magnetic_field[face_element_number, face_nodes, :])
            boundary_view.equilibrium.magnetic_field_unit[key].append(glob_view.equilibrium.magnetic_field_unit[face_element_number, face_nodes, :])
            boundary_view.equilibrium.poloidal_flux[key].append(glob_view.equilibrium.poloidal_flux[face_element_number, face_nodes])

    if solution.n_partitions > 1:
        solution_skeleton_boundary = np.ones((solution.mesh.nfaces_glob, solution.mesh.mesh_parameters["nodes_per_face"], solution.neq))
        for i in range(solution.n_partitions):
            non_ghost = ~solution.mesh.raw_ghost_faces[i].flatten()
            raw_solution = solution.raw.solutions_skeleton[i].reshape(
                solution.raw.solutions_skeleton[i].shape[0] // solution.mesh.mesh_parameters["nodes_per_face"],
                solution.mesh.mesh_parameters["nodes_per_face"],
                solution.neq,
            )
            solution_skeleton_boundary[solution.mesh.raw_rest_mesh_data[i]["loc2glob_fa"][:][non_ghost], :] = raw_solution[non_ghost]
        solution_skeleton_boundary = solution_skeleton_boundary[solution.mesh.boundary_state.filled, :, :]
    else:
        solution_skeleton_boundary = solution.raw.solutions_skeleton[0].reshape(
            solution.raw.solutions_skeleton[0].shape[0] // solution.mesh.mesh_parameters["nodes_per_face"],
            solution.mesh.mesh_parameters["nodes_per_face"],
            solution.neq,
        )
        solution_skeleton_boundary = solution_skeleton_boundary[-solution.mesh.raw_mesh_numbers[0]["Nextfaces"] :, :, :]
    print(solution_skeleton_boundary.shape)
    for key, indices in solution.mesh.boundary_state.indices.items():
        boundary_view.solution_skeleton.conservative[key] = []
        for ind in indices:
            boundary_view.solution_skeleton.conservative[key].append(solution_skeleton_boundary[ind, :, :])

    solution.metadata.flags.combined_boundary = True
    solution.metadata.flags.combined_boundary_gauss = False
    solution.metadata.cache.boundary_gauss_boundaries = None
    solution.metadata.cache.boundary_gauss_ordering = None
    solution.metadata.cache.boundary_gauss_connectivity = None
    solution.metadata.cache.boundary_gauss_face_elements = None


def calculate_in_gauss_points(solution):
    if not solution.metadata.flags.combined_to_full:
        print("Comibining first solution full")
        recombine_full_solution(solution)
    if solution.mesh.metadata.reference_element is None:
        raise ValueError("Please, provide reference element to the mesh")

    glob_view = solution.views.glob
    gauss_view = solution.views.gauss
    gauss_view.solution.conservative = np.einsum("ij,kjh->kih", solution.mesh.metadata.reference_element["N"], glob_view.solution.conservative)
    gauss_view.gradient.conservative = np.einsum("ij,kjhl->kihl", solution.mesh.metadata.reference_element["N"], glob_view.gradient.conservative)
    gauss_view.equilibrium.magnetic_field = np.einsum("ij,kjh->kih", solution.mesh.metadata.reference_element["N"], glob_view.equilibrium.magnetic_field)
    gauss_view.equilibrium.magnetic_field_unit = np.einsum(
        "ij,kjh->kih", solution.mesh.metadata.reference_element["N"], glob_view.equilibrium.magnetic_field_unit
    )
    gauss_view.equilibrium.jtor = np.einsum("ij,kj->ki", solution.mesh.metadata.reference_element["N"], glob_view.equilibrium.jtor)
    gauss_view.equilibrium.poloidal_flux = np.einsum("ij,kj->ki", solution.mesh.metadata.reference_element["N"], glob_view.equilibrium.poloidal_flux)

    if "external_heating" in solution.parameters["physics"]:
        gauss_view.sources.external_heating = np.einsum("ij,kj->ki", solution.mesh.metadata.reference_element["N"], solution.views.glob.sources.external_heating)
    if "external_heating_e" in solution.parameters["physics"]:
        gauss_view.sources.external_heating_e = np.einsum("ij,kj->ki", solution.mesh.metadata.reference_element["N"], solution.views.glob.sources.external_heating_e)
    if "external_heating_i" in solution.parameters["physics"]:
        gauss_view.sources.external_heating_i = np.einsum("ij,kj->ki", solution.mesh.metadata.reference_element["N"], solution.views.glob.sources.external_heating_i)
    solution.metadata.flags.combined_gauss = True
    solution.metadata.flags.gauss_phys_initialized = False


def calculate_in_boundary_gauss_points(solution, boundaries):
    if solution.mesh.metadata.reference_element is None:
        raise ValueError("Please, provide reference element to the mesh")
    if not solution.metadata.flags.combined_boundary:
        print("Comibining first values on boundary")
        recombine_boundary_solution(solution)
    normalized_boundaries = tuple(np.asarray(boundaries, dtype=int).tolist())
    cache = solution.metadata.cache
    if solution.metadata.flags.combined_boundary_gauss and cache.boundary_gauss_boundaries == normalized_boundaries:
        return (
            cache.boundary_gauss_ordering,
            cache.boundary_gauss_connectivity,
            cache.boundary_gauss_face_elements,
        )
    boundary_ordering, connectivity_ordered, iel_face_ordered = solution.mesh.boundary.compute_gauss(
        normalized_boundaries, solution.raw.boundary_infos
    )
    boundary_view = solution.views.boundary
    boundary_solution = boundary_view.solution.conservative
    boundary_solution_skeleton = boundary_view.solution_skeleton.conservative
    boundary_gradient = boundary_view.gradient.conservative
    boundary_equilibrium = boundary_view.equilibrium

    solution_boundary_ordered = np.empty(
        [0, boundary_solution[normalized_boundaries[0]][0].shape[1], boundary_solution[normalized_boundaries[0]][0].shape[2]]
    )
    solution_skeleton_boundary_ordered = np.empty(
        [0, boundary_solution_skeleton[normalized_boundaries[0]][0].shape[1], boundary_solution_skeleton[normalized_boundaries[0]][0].shape[2]]
    )
    gradient_boundary_ordered = np.empty(
        [0, boundary_gradient[normalized_boundaries[0]][0].shape[1], boundary_gradient[normalized_boundaries[0]][0].shape[2], boundary_gradient[normalized_boundaries[0]][0].shape[3]]
    )
    magnetic_field_boundary_ordered = np.empty(
        [0, boundary_equilibrium.magnetic_field[normalized_boundaries[0]][0].shape[1], boundary_equilibrium.magnetic_field[normalized_boundaries[0]][0].shape[2]]
    )
    magnetic_field_unit_boundary_ordered = np.empty(
        [0, boundary_equilibrium.magnetic_field_unit[normalized_boundaries[0]][0].shape[1], boundary_equilibrium.magnetic_field_unit[normalized_boundaries[0]][0].shape[2]]
    )
    poloidal_flux_boundary_ordered = np.empty([0, boundary_equilibrium.poloidal_flux[normalized_boundaries[0]][0].shape[1]])
    for bound_order in boundary_ordering:
        solution_boundary_ordered = np.vstack([solution_boundary_ordered, boundary_solution[normalized_boundaries[bound_order[0]]][bound_order[1]]])
        solution_skeleton_boundary_ordered = np.vstack([
            solution_skeleton_boundary_ordered,
            boundary_solution_skeleton[normalized_boundaries[bound_order[0]]][bound_order[1]],
        ])
        gradient_boundary_ordered = np.vstack([gradient_boundary_ordered, boundary_gradient[normalized_boundaries[bound_order[0]]][bound_order[1]]])
        magnetic_field_boundary_ordered = np.vstack([magnetic_field_boundary_ordered, boundary_equilibrium.magnetic_field[normalized_boundaries[bound_order[0]]][bound_order[1]]])
        magnetic_field_unit_boundary_ordered = np.vstack([
            magnetic_field_unit_boundary_ordered,
            boundary_equilibrium.magnetic_field_unit[normalized_boundaries[bound_order[0]]][bound_order[1]],
        ])
        poloidal_flux_boundary_ordered = np.vstack([poloidal_flux_boundary_ordered, boundary_equilibrium.poloidal_flux[normalized_boundaries[bound_order[0]]][bound_order[1]]])
    boundary_gauss_view = solution.views.boundary_gauss
    boundary_gauss_view.solution.conservative = np.einsum("ij,kjh->kih", solution.mesh.metadata.reference_element["N1d"], solution_boundary_ordered)
    boundary_gauss_view.solution_skeleton.conservative = np.einsum("ij,kjh->kih", solution.mesh.metadata.reference_element["N1d"], solution_skeleton_boundary_ordered)
    boundary_gauss_view.gradient.conservative = np.einsum("ij,kjhl->kihl", solution.mesh.metadata.reference_element["N1d"], gradient_boundary_ordered)
    boundary_gauss_view.equilibrium.magnetic_field = np.einsum("ij,kjh->kih", solution.mesh.metadata.reference_element["N1d"], magnetic_field_boundary_ordered)
    boundary_gauss_view.equilibrium.poloidal_flux = np.einsum("ij,kj->ki", solution.mesh.metadata.reference_element["N1d"], poloidal_flux_boundary_ordered)
    boundary_gauss_view.equilibrium.magnetic_field_unit = np.einsum(
        "ij,kjh->kih", solution.mesh.metadata.reference_element["N1d"], magnetic_field_unit_boundary_ordered
    )
    solution.metadata.flags.combined_boundary_gauss = True
    cache.boundary_gauss_boundaries = normalized_boundaries
    cache.boundary_gauss_ordering = boundary_ordering
    cache.boundary_gauss_connectivity = connectivity_ordered
    cache.boundary_gauss_face_elements = iel_face_ordered
    return boundary_ordering, connectivity_ordered, iel_face_ordered
