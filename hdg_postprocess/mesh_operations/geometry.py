import os
from pathlib import Path

import numpy as np
from raysect.core.math.function.float import Discrete2DMesh


def recombine_full_mesh(mesh):
    mesh._nelems_glob = 0
    mesh._nvertices_glob = 0

    for i in range(mesh.n_partitions):
        mesh._nelems_glob = max(mesh._nelems_glob, mesh.raw_rest_mesh_data[i]["loc2glob_el"].max())
        mesh._nvertices_glob = max(mesh._nvertices_glob, mesh.raw_rest_mesh_data[i]["loc2glob_no"].max())
    mesh._nelems_glob += 1
    mesh._nvertices_glob += 1
    mesh._connectivity_glob = np.zeros((mesh._nelems_glob, mesh.mesh_parameters["nodes_per_element"]), dtype=int)
    mesh._vertices_glob = np.zeros((mesh._nvertices_glob, mesh.mesh_parameters["Ndim"]))
    for i in range(mesh.n_partitions):
        mesh._connectivity_glob[mesh.raw_rest_mesh_data[i]["loc2glob_el"][~mesh.raw_ghost_elements[i].astype(bool).flatten()]] = (
            mesh.raw_rest_mesh_data[i]["loc2glob_no"][mesh.raw_connectivity[i][~mesh.raw_ghost_elements[i].astype(bool).flatten(), :]]
        )
        mesh._vertices_glob[mesh.raw_rest_mesh_data[i]["loc2glob_no"], :] = mesh.raw_vertices[i][:, :]
    mesh._combined_to_full = True


def create_connectivity_big(mesh):
    if not mesh._combined_to_full:
        recombine_full_mesh(mesh)
    base_path = Path(__file__).resolve().parents[1]
    rel_path = f'data/triangulations_element/{mesh.mesh_parameters["element_type"]}_P{mesh.p_order}.npy'
    path = (base_path / rel_path).resolve()
    if not os.path.isfile(path):
        raise KeyError(f"{path} splitting of each element is not defined yet")

    triangle_indexes = np.load(path)
    connectivity_big = mesh.connectivity_glob[:, triangle_indexes]
    mesh._connectivity_big = connectivity_big.reshape(connectivity_big.shape[0] * connectivity_big.shape[1], 3)


def make_mask(mesh):
    if mesh.connectivity_big is None:
        create_connectivity_big(mesh)

    mesh._mask = Discrete2DMesh(
        mesh.vertices_glob, mesh.connectivity_big, np.ones(mesh.connectivity_big.shape[0]), limit=False, default_value=0
    )


def make_element_number_function(mesh):
    if mesh.connectivity_big is None:
        create_connectivity_big(mesh)
    element_numbers = np.repeat(
        np.arange(len(mesh.connectivity_glob)),
        mesh.connectivity_big.shape[0] // mesh.connectivity_glob.shape[0],
    )
    mesh._element_number = Discrete2DMesh(
        mesh.vertices_glob, mesh.connectivity_big, element_numbers, limit=False, default_value=-1
    )


def calculate_gauss_volumes(mesh):
    if mesh.reference_element is None:
        raise ValueError("Please, provide reference element")
    if not mesh._combined_to_full:
        recombine_full_mesh(mesh)

    mesh._vertices_gauss = np.einsum("ij,kjh->kih", mesh.reference_element["N"], mesh.vertices_glob[mesh.connectivity_glob, :])
    J11_loc = np.einsum("ij,kj->ki", mesh.reference_element["Nxi"], mesh.vertices_glob[mesh.connectivity_glob, 0])
    J12_loc = np.einsum("ij,kj->ki", mesh.reference_element["Nxi"], mesh.vertices_glob[mesh.connectivity_glob, 1])
    J21_loc = np.einsum("ij,kj->ki", mesh.reference_element["Neta"], mesh.vertices_glob[mesh.connectivity_glob, 0])
    J22_loc = np.einsum("ij,kj->ki", mesh.reference_element["Neta"], mesh.vertices_glob[mesh.connectivity_glob, 1])
    detJ_loc = J11_loc * J22_loc - J12_loc * J21_loc

    mesh._volumes_gauss = 2 * np.pi * mesh.reference_element["IPweights"][None, :] * detJ_loc[:, :] * mesh._vertices_gauss[:, :, 0]


def find_adjacent_elements(mesh, element_number):
    if not mesh._combined_to_full:
        print("Comibining to full mesh")
        recombine_full_mesh(mesh)
    vertices_numbers = mesh.connectivity_glob[element_number]

    adjacent_numbers = []
    for number in vertices_numbers:
        idx = np.where(number == mesh.connectivity_glob)
        for i in idx[0]:
            if i != element_number and i not in adjacent_numbers:
                adjacent_numbers.append(i)
    return adjacent_numbers
