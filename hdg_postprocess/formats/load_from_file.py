from hdg_postprocess.HDG_solution import HDGsolution
from hdg_postprocess.HDG_mesh import HDGmesh
from hdg_postprocess.io import load_mesh_data, load_solution_data


def load_HDG_solution_from_file(solpath,solname_base,meshpath=None,meshname_base=None,n_partitions=1):
    """
    Load an SOLEDGE-HDG simulation from SOLEDGE-HDG file(s) and meshile(s).
    :param str solpath: String path to a folder contatinging file(s) with simulation SOLEDGE-HDG.
    :param str solname_base: String base of the name of the solution (without .h5 and numerings of the partitions)
    :param str meshpath: String path to a folder contatinging file(s) with mesh(es) SOLEDGE-HDG.
    :param str meshname_base:tring base of the name of the mesh (without .h5 and numerings of the partitions)
    :param int n_partitions: number of partitions used in the simulation
    :rtype: HDGsolution
    """
    solution_data = load_solution_data(solpath, solname_base, meshpath, meshname_base, n_partitions)
    mesh = load_HDG_mesh_from_file(solution_data.mesh_path, solution_data.mesh_name_base, n_partitions)

    sol = HDGsolution(
        solution_data.raw_solutions,
        solution_data.raw_solutions_skeleton,
        solution_data.raw_gradients,
        solution_data.raw_equilibriums,
        solution_data.raw_solution_boundary_infos,
        solution_data.parameters,
        solution_data.n_partitions,
        mesh,
        raw_transport_1d=solution_data.raw_transport_1d,
        raw_neutral_flux_limiter_diagnostics=solution_data.raw_neutral_flux_limiter_diagnostics,
    )
    
    return sol

def load_HDG_mesh_from_file(meshpath,meshname_base,n_partitions):
    """
    Load an SOLEDGE-HDG simulation from SOLEDGE-HDG file(s) and meshile(s).
    :param str meshpath: String path to a folder contatinging file(s) with mesh(es) SOLEDGE-HDG.
    :param str meshname_base:tring base of the name of the mesh (without .h5 and numerings of the partitions)
    :param int n_partitions: number of partitions used in the simulation
    :rtype: HDGmesh
    """

    mesh_data = load_mesh_data(meshpath, meshname_base, n_partitions)
    mesh = HDGmesh(
        mesh_data.raw_vertices,
        mesh_data.raw_connectivity,
        mesh_data.raw_connectivity_boundary,
        mesh_data.raw_mesh_numbers,
        mesh_data.raw_boundary_flags,
        mesh_data.raw_ghost_elements,
        mesh_data.raw_ghost_faces,
        mesh_data.mesh_parameters,
        mesh_data.n_partitions,
        mesh_data.raw_rest_mesh_data,
    )
    return mesh
