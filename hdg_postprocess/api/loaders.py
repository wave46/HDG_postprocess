def load_solution(solpath, solname_base, meshpath=None, meshname_base=None, n_partitions=1):
    """Load a solution using the structured HDGsolution API."""
    from hdg_postprocess.formats import load_HDG_solution_from_file

    return load_HDG_solution_from_file(solpath, solname_base, meshpath, meshname_base, n_partitions)


def load_mesh(meshpath, meshname_base, n_partitions=1):
    """Load a mesh using the structured HDGmesh API."""
    from hdg_postprocess.formats import load_HDG_mesh_from_file

    return load_HDG_mesh_from_file(meshpath, meshname_base, n_partitions)
