from copy import deepcopy

from silx.io.dictdump import h5todict

from .types import NormalizedMeshData, NormalizedSolutionData


def _build_partition_filename(path, name_base, n_partition, n_partitions):
    if n_partitions == 1:
        return f"{path}{name_base}.h5"
    return f"{path}{name_base}_{n_partition}_{n_partitions}.h5"


def _normalize_simulation_parameters(parameters):
    normalized = deepcopy(parameters)
    for key, item in normalized["adimensionalization"].items():
        if len(item) == 1:
            normalized["adimensionalization"][key] = item[0]
    for key, item in normalized["physics"].items():
        if key == "atomic":
            normalized["physics"][key] = item
        elif len(item) == 1:
            normalized["physics"][key] = item[0]
    return normalized


def detect_solution_format(solution_file):
    if "solution" in solution_file:
        return "grouped_solution"
    return "flat_solution"


def detect_mesh_format(mesh_file):
    if "mesh" in mesh_file:
        return "grouped_mesh"
    return "flat_mesh"


def _extract_solution_partition(solution_file, parameters):
    solution_group = solution_file["solution"] if detect_solution_format(solution_file) == "grouped_solution" else solution_file
    magnetic_group = solution_file["magnetic"] if "magnetic" in solution_file else solution_file
    boundary_group = solution_file["mesh"] if "mesh" in solution_file else solution_file

    equilibrium = {}
    if parameters["switches"]["ohmicsrc"][0] == 1:
        equilibrium["plasma_current"] = magnetic_group["Jtor"].T
    equilibrium["magnetic_field"] = magnetic_group["magnetic_field"].T
    if "magnetic_psi" in magnetic_group:
        equilibrium["poloidal_flux"] = magnetic_group["magnetic_psi"].T

    solution_boundary_data = {}
    if "boundary_flags" in boundary_group:
        solution_boundary_data["boundary_flags"] = boundary_group["boundary_flags"].T
    elif "boundaryFlag" in boundary_group:
        solution_boundary_data["boundary_flags"] = boundary_group["boundaryFlag"].T

    if "exterior_faces" in boundary_group:
        solution_boundary_data["exterior_faces"] = boundary_group["exterior_faces"].T.astype(int) - 1
    elif "extfaces" in boundary_group:
        solution_boundary_data["exterior_faces"] = boundary_group["extfaces"].T.astype(int) - 1

    transport_1d = {}
    transport_group = None
    if "transport_1d" in solution_file:
        transport_group = solution_file["transport_1d"]
    elif isinstance(solution_group, dict) and "transport_1d" in solution_group:
        transport_group = solution_group["transport_1d"]
    if transport_group is not None:
        for key, value in transport_group.items():
            transport_1d[key] = value

    return {
        "raw_solution": solution_group["u"],
        "raw_solution_skeleton": solution_group["u_tilde"],
        "raw_gradient": solution_group["q"],
        "equilibrium": equilibrium,
        "boundary_info": solution_boundary_data,
        "transport_1d": transport_1d,
    }


def load_solution_data(solpath, solname_base, meshpath=None, meshname_base=None, n_partitions=1):
    raw_solutions = []
    raw_solutions_skeleton = []
    raw_gradients = []
    raw_equilibriums = []
    raw_solution_boundary_infos = []
    raw_transport_1d = []

    parameters = None

    for n_partition in range(1, n_partitions + 1):
        solfile_name = _build_partition_filename(solpath, solname_base, n_partition, n_partitions)
        solution_file = h5todict(solfile_name)

        if n_partition == 1:
            parameters = _normalize_simulation_parameters(solution_file["simulation_parameters"])

        partition_data = _extract_solution_partition(solution_file, parameters)
        raw_solutions.append(partition_data["raw_solution"])
        raw_solutions_skeleton.append(partition_data["raw_solution_skeleton"])
        raw_gradients.append(partition_data["raw_gradient"])
        raw_equilibriums.append(partition_data["equilibrium"])
        raw_solution_boundary_infos.append(partition_data["boundary_info"])
        raw_transport_1d.append(partition_data["transport_1d"])

    if meshpath is None and meshname_base is None:
        meshpath = solpath
        meshname_base = solname_base
    elif meshpath is None or meshname_base is None:
        raise ValueError("If meshpath or meshname_base is given, both must be given")

    return NormalizedSolutionData(
        raw_solutions=raw_solutions,
        raw_solutions_skeleton=raw_solutions_skeleton,
        raw_gradients=raw_gradients,
        raw_equilibriums=raw_equilibriums,
        raw_solution_boundary_infos=raw_solution_boundary_infos,
        raw_transport_1d=raw_transport_1d,
        parameters=parameters,
        n_partitions=n_partitions,
        mesh_path=meshpath,
        mesh_name_base=meshname_base,
    )


def _extract_mesh_partition(mesh_file, n_partition, n_partitions):
    mesh_group = mesh_file["mesh"] if detect_mesh_format(mesh_file) == "grouped_mesh" else mesh_file

    mesh_partition = {
        "vertices": mesh_group["X"].T.copy(order="C"),
        "connectivity": mesh_group["T"].T.astype(int) - 1,
        "connectivity_boundary": mesh_group["Tb"].T.astype(int) - 1,
        "mesh_numbers": {
            "Nelems": int(mesh_group["Nelems"][0]),
            "Nextfaces": int(mesh_group["Nextfaces"][0]),
            "Nnodes": int(mesh_group["Nnodes"][0]),
        },
    }

    if n_partitions > 1:
        mesh_partition["mesh_numbers"]["Nfaces"] = int(mesh_group["Nfaces"][0])
        mesh_partition["rest_mesh_data"] = {
            "ghelsLoc": mesh_group["ghelsLoc"],
            "ghelsPro": mesh_group["ghelsPro"],
            "ghostFlp": mesh_group["ghostFlp"],
            "ghostLoc": mesh_group["ghostLoc"],
            "ghostPro": mesh_group["ghostPro"],
            "loc2glob_el": mesh_group["loc2glob_el"].T.astype(int) - 1,
            "loc2glob_fa": mesh_group["loc2glob_fa"].T.astype(int) - 1,
            "loc2glob_no": mesh_group["loc2glob_no"].T.astype(int) - 1,
        }
        mesh_partition["boundary_flags"] = mesh_group["boundaryFlag"].T.astype(int)
        mesh_partition["ghost_elements"] = mesh_group["ghostElems"][:, None].astype(bool)
        mesh_partition["ghost_faces"] = mesh_group["ghostFaces"][:, None].astype(bool)

    if n_partition == 1:
        mesh_partition["mesh_parameters"] = {
            "Ndim": int(mesh_group["Ndim"][0]),
            "nodes_per_element": int(mesh_group["Nnodesperelem"][0]),
            "nodes_per_face": int(mesh_group["Nnodesperface"][0]),
            "element_type": "triangle" if mesh_group["elemType"][0] == 0 else "quadrilateral",
        }
        if "elemSize" in mesh_group:
            mesh_partition["mesh_parameters"]["elemSize"] = mesh_group["elemSize"]

    return mesh_partition


def load_mesh_data(meshpath, meshname_base, n_partitions):
    mesh_parameters = None
    raw_vertices = []
    raw_connectivity = []
    raw_connectivity_boundary = []
    raw_mesh_numbers = []
    raw_boundary_flags = []
    raw_ghost_elements = []
    raw_ghost_faces = []
    raw_rest_mesh_data = []

    for n_partition in range(1, n_partitions + 1):
        meshfile_name = _build_partition_filename(meshpath, meshname_base, n_partition, n_partitions)
        mesh_file = h5todict(meshfile_name)
        partition_data = _extract_mesh_partition(mesh_file, n_partition, n_partitions)

        if n_partition == 1:
            mesh_parameters = partition_data["mesh_parameters"]

        raw_vertices.append(partition_data["vertices"])
        raw_connectivity.append(partition_data["connectivity"])
        raw_connectivity_boundary.append(partition_data["connectivity_boundary"])
        raw_mesh_numbers.append(partition_data["mesh_numbers"])

        if n_partitions > 1:
            raw_rest_mesh_data.append(partition_data["rest_mesh_data"])
            raw_boundary_flags.append(partition_data["boundary_flags"])
            raw_ghost_elements.append(partition_data["ghost_elements"])
            raw_ghost_faces.append(partition_data["ghost_faces"])

    return NormalizedMeshData(
        raw_vertices=raw_vertices,
        raw_connectivity=raw_connectivity,
        raw_connectivity_boundary=raw_connectivity_boundary,
        raw_mesh_numbers=raw_mesh_numbers,
        raw_boundary_flags=raw_boundary_flags,
        raw_ghost_elements=raw_ghost_elements,
        raw_ghost_faces=raw_ghost_faces,
        mesh_parameters=mesh_parameters,
        n_partitions=n_partitions,
        raw_rest_mesh_data=raw_rest_mesh_data,
    )
