from hdg_postprocess.HDG_solution import HDGsolution
from hdg_postprocess.HDG_mesh import HDGmesh
from silx.io.dictdump import h5todict
import numpy as np


def load_HDG_solution_from_file(solpath,solname_base,meshpath,meshname_base,n_partitions):
    """
    Load an SOLEDGE-HDG simulation from SOLEDGE-HDG file(s) and meshile(s).
    :param str solpath: String path to a folder contatinging file(s) with simulation SOLEDGE-HDG.
    :param str solname_base: String base of the name of the solution (without .h5 and numerings of the partitions)
    :param str meshpath: String path to a folder contatinging file(s) with mesh(es) SOLEDGE-HDG.
    :param str meshname_base:tring base of the name of the mesh (without .h5 and numerings of the partitions)
    :param int n_partitions: number of partitions used in the simulation
    :rtype: HDGsolution
    """
    # lists for the solution files
    raw_solutions = []                                  #raw solutions "u" for every node for every element as it comes from HDG
    raw_solutions_skeleton = []                         #raw solutions "u_tilde" on a skeleton mesh as it comes from HDG
    raw_gradients = []                                  #raw gradients "q" for every node for every element as it comes from HDG
    raw_equilibriums = []                                #list with dictionaries with plasma current, magnetic field, poloidal flux
    raw_solution_boundary_infos= []                     #list with dictionaries with info about boundary flags and exterior faces (not sure if it will be needed later)
    meshpaths,solpaths = [],[]
    parameters = {}                                     #parameters of the simulation which do not change between partitions
    
    #lists for the mesh data
    mesh_parameters = {}                                #parameters of the mesh which do not change between partitions
    raw_vertices = []
    raw_connectivity = []
    raw_connectivity_boundary = []
    raw_mesh_numbers = []                               #list containing dictionaries with number of elements, external faces, faces, nodes in the given mesh
    raw_boundary_flags = []                             #list with boundary flags to_add: boundary flags description
    raw_ghost_elements = []                             #list with flag if ghost element, then True, if not ghost then False
    raw_ghost_faces = []                                #list with flag if ghost face, then True, if not ghost then False
    raw_rest_mesh_data = []                             #list with the rest mesh data not well understood and used so far
    for n_partition in range(1,n_partitions+1): 
        # define the name of what is going to be read
        if n_partitions == 1:
            #if there is 1 partition used, no 1_8 ending in the filename        
            solfile_name = f'{solpath}{solname_base}.h5'
            meshfile_name = f'{meshpath}{meshname_base}.h5'
        else:
            solfile_name = f'{solpath}{solname_base}_{n_partition}_{n_partitions}.h5'
            meshfile_name = f'{meshpath}{meshname_base}_{n_partition}_{n_partitions}.h5'
        
        #reading solution file
        solution_file = h5todict(solfile_name)

        if n_partition == 1:
            parameters= solution_file['simulation_parameters']          #save all the simulation parameters
            for key,item in parameters['adimensionalization'].items():
                if len(item)==1:
                    parameters['adimensionalization'][key] = item[0]
            for key,item in parameters['physics'].items():
                if len(item)==1:
                    parameters['physics'][key] = item[0]
        
        # stack raw data
        raw_solutions.append(solution_file['u'])
        raw_solutions_skeleton.append(solution_file['u_tilde'])
        raw_gradients.append(solution_file['q'])

        #define equilibrium dictionary
        equilibrium = {}
        equilibrium['plasma_current'] = solution_file['Jtor'].T
        equilibrium['magnetic_field'] = solution_file['magnetic_field'].T
        equilibrium['poloidal_flux'] = solution_file['magnetic_psi'].T   #this might be corrupted or normalized
        raw_equilibriums.append(equilibrium)

        #define boundary dictionary
        solution_boundary_data = {}
        solution_boundary_data['boundary_flags'] = solution_file['boundary_flags'].T
        solution_boundary_data['exterior_faces'] = solution_file['exterior_faces'].T.astype(int) -1
        raw_solution_boundary_infos.append(solution_boundary_data)

        #reading mesh file
        mesh_file = h5todict(meshfile_name)
        if n_partition == 1:
            mesh_parameters['Ndim'] = int(mesh_file['Ndim'][0])
            mesh_parameters['nodes_per_element'] = int(mesh_file['Nnodesperelem'][0])
            mesh_parameters['nodes_per_face'] = int(mesh_file['Nnodesperface'][0])
            if mesh_file['elemType'][0] == 0:
                mesh_parameters['element_type'] = 'triangle'
            else:
                mesh_parameters['element_type'] = 'quadrangle'

        raw_vertices.append((mesh_file['X'].T).copy(order='C'))
        raw_connectivity.append(mesh_file['T'].T.astype(int) -1)
        raw_connectivity_boundary.append(mesh_file['Tb'].T.astype(int) -1)

        mesh_numbers = {}
        mesh_numbers['Nelems'] = int(mesh_file['Nelems'][0])
        mesh_numbers['Nextfaces'] = int(mesh_file['Nextfaces'][0])
        mesh_numbers['Nfaces'] = int(mesh_file['Nfaces'][0])
        mesh_numbers['Nnodes'] = int(mesh_file['Nnodes'][0])
        raw_mesh_numbers.append(mesh_numbers)

        raw_boundary_flags.append(mesh_file['boundaryFlag'].T.astype(int))
        raw_ghost_elements.append(mesh_file['ghostElems'][:,None].astype(bool))
        raw_ghost_faces.append(mesh_file['ghostFaces'][:,None].astype(bool))

        meshpaths.append(meshfile_name)
        solpaths.append(solfile_name)

        rest_mesh_data = {}
        rest_mesh_data['ghelsLoc'] = mesh_file['ghelsLoc']
        rest_mesh_data['ghelsPro'] = mesh_file['ghelsPro']
        rest_mesh_data['ghostFlp'] = mesh_file['ghostFlp']
        rest_mesh_data['ghostLoc'] = mesh_file['ghostLoc']
        rest_mesh_data['ghostPro'] = mesh_file['ghostPro']
        rest_mesh_data['loc2glob_el'] = mesh_file['loc2glob_el'].T.astype(int) -1
        rest_mesh_data['loc2glob_fa'] = mesh_file['loc2glob_fa'].T.astype(int) -1
        rest_mesh_data['loc2glob_no'] = mesh_file['loc2glob_no'].T.astype(int) -1
        raw_rest_mesh_data.append(rest_mesh_data)
    mesh = HDGmesh(raw_vertices,raw_connectivity,raw_connectivity_boundary,
                raw_mesh_numbers,raw_boundary_flags,raw_ghost_elements,
                raw_ghost_faces,mesh_parameters,n_partitions,raw_rest_mesh_data)
    sol = HDGsolution(raw_solutions, raw_solutions_skeleton, raw_gradients,
                 raw_equilibriums,raw_solution_boundary_infos, parameters, 
                 n_partitions, mesh)
    
    return sol

def load_HDG_mesh_from_file(meshpath,meshname_base,n_partitions):
    """
    Load an SOLEDGE-HDG simulation from SOLEDGE-HDG file(s) and meshile(s).
    :param str meshpath: String path to a folder contatinging file(s) with mesh(es) SOLEDGE-HDG.
    :param str meshname_base:tring base of the name of the mesh (without .h5 and numerings of the partitions)
    :param int n_partitions: number of partitions used in the simulation
    :rtype: HDGmesh
    """

    #lists for the mesh data
    mesh_parameters = {}                                #parameters of the mesh which do not change between partitions
    raw_vertices = []
    raw_connectivity = []
    raw_connectivity_boundary = []
    raw_mesh_numbers = []                               #list containing dictionaries with number of elements, external faces, faces, nodes in the given mesh
    raw_boundary_flags = []                             #list with boundary flags to_add: boundary flags description
    raw_ghost_elements = []                             #list with flag if ghost element, then True, if not ghost then False
    raw_ghost_faces = []                                #list with flag if ghost face, then True, if not ghost then False
    raw_rest_mesh_data = []                             #list with the rest mesh data not well understood and used so far

    for n_partition in range(1,n_partitions+1): 
        # define the name of what is going to be read
        if n_partitions == 1:
            #if there is 1 partition used, no 1_8 ending in the filename        
            meshfile_name = f'{meshpath}{meshname_base}.h5'
        else:
            meshfile_name = f'{meshpath}{meshname_base}_{n_partition}_{n_partitions}.h5'

    #reading mesh file
        mesh_file = h5todict(meshfile_name)
        if n_partition == 1:
            mesh_parameters['Ndim'] = int(mesh_file['Ndim'][0])
            mesh_parameters['nodes_per_element'] = int(mesh_file['Nnodesperelem'][0])
            mesh_parameters['nodes_per_face'] = int(mesh_file['Nnodesperface'][0])
            if mesh_file['elemType'][0] == 0:
                mesh_parameters['element_type'] = 'triangle'
            else:
                mesh_parameters['element_type'] = 'quadrangle'

        raw_vertices.append((mesh_file['X'].T).copy(order='C'))
        raw_connectivity.append(mesh_file['T'].T.astype(int) -1)
        raw_connectivity_boundary.append(mesh_file['Tb'].T.astype(int) -1)

        mesh_numbers = {}
        mesh_numbers['Nelems'] = int(mesh_file['Nelems'][0])
        mesh_numbers['Nextfaces'] = int(mesh_file['Nextfaces'][0])
        mesh_numbers['Nfaces'] = int(mesh_file['Nfaces'][0])
        mesh_numbers['Nnodes'] = int(mesh_file['Nnodes'][0])
        raw_mesh_numbers.append(mesh_numbers)

        raw_boundary_flags.append(mesh_file['boundaryFlag'].T.astype(int))
        raw_ghost_elements.append(mesh_file['ghostElems'][:,None].astype(bool))
        raw_ghost_faces.append(mesh_file['ghostFaces'][:,None].astype(bool))
        if n_partitions>1:
            #this info for parallel version
            rest_mesh_data = {}
            rest_mesh_data['ghelsLoc'] = mesh_file['ghelsLoc']
            rest_mesh_data['ghelsPro'] = mesh_file['ghelsPro']
            rest_mesh_data['ghostFlp'] = mesh_file['ghostFlp']
            rest_mesh_data['ghostLoc'] = mesh_file['ghostLoc']
            rest_mesh_data['ghostPro'] = mesh_file['ghostPro']
            rest_mesh_data['loc2glob_el'] = mesh_file['loc2glob_el'].T.astype(int) -1
            rest_mesh_data['loc2glob_fa'] = mesh_file['loc2glob_fa'].T.astype(int) -1
            rest_mesh_data['loc2glob_no'] = mesh_file['loc2glob_no'].T.astype(int) -1
            raw_rest_mesh_data.append(rest_mesh_data)

    mesh = HDGmesh(raw_vertices,raw_connectivity,raw_connectivity_boundary,
                raw_mesh_numbers,raw_boundary_flags,raw_ghost_elements,
                raw_ghost_faces,mesh_parameters,n_partitions,raw_rest_mesh_data)
    return mesh