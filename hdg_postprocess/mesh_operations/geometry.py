import os
from pathlib import Path

import numpy as np
from raysect.core.math.function.float import Discrete2DMesh
from hdg_postprocess.mesh_operations import assembly as assembly_ops


def _ensure_full_mesh(mesh):
    if not mesh.metadata.flags.combined_to_full:
        assembly_ops.recombine_full_mesh(mesh)


def _ensure_connectivity_big(mesh):
    if not mesh.metadata.flags.connectivity_big_initialized:
        create_connectivity_big(mesh)


def create_connectivity_big(mesh):
    _ensure_full_mesh(mesh)
    base_path = Path(__file__).resolve().parents[1]
    rel_path = f'data/triangulations_element/{mesh.mesh_parameters["element_type"]}_P{mesh.metadata.p_order}.npy'
    path = (base_path / rel_path).resolve()
    if not os.path.isfile(path):
        raise KeyError(f"{path} splitting of each element is not defined yet")

    triangle_indexes = np.load(path)
    connectivity_big = mesh.global_state.connectivity[:, triangle_indexes]
    mesh.derived_geometry.connectivity_big = connectivity_big.reshape(connectivity_big.shape[0] * connectivity_big.shape[1], 3)
    mesh.metadata.flags.connectivity_big_initialized = True


def make_mask(mesh):
    _ensure_connectivity_big(mesh)

    mesh.derived_geometry.mask = Discrete2DMesh(
        mesh.global_state.vertices, mesh.derived_geometry.connectivity_big, np.ones(mesh.derived_geometry.connectivity_big.shape[0]), limit=False, default_value=0
    )
    mesh.metadata.flags.mask_initialized = True


def make_element_number_function(mesh):
    _ensure_connectivity_big(mesh)
    element_numbers = np.repeat(
        np.arange(len(mesh.global_state.connectivity)),
        mesh.derived_geometry.connectivity_big.shape[0] // mesh.global_state.connectivity.shape[0],
    )
    mesh.derived_geometry.element_locator = Discrete2DMesh(
        mesh.global_state.vertices, mesh.derived_geometry.connectivity_big, element_numbers, limit=False, default_value=-1
    )
    mesh.metadata.flags.element_locator_initialized = True


def calculate_gauss_volumes(mesh):
    if mesh.metadata.reference_element is None:
        raise ValueError("Please, provide reference element")
    _ensure_full_mesh(mesh)

    mesh.derived_geometry.vertices_gauss = np.einsum("ij,kjh->kih", mesh.metadata.reference_element["N"], mesh.global_state.vertices[mesh.global_state.connectivity, :])
    J11_loc = np.einsum("ij,kj->ki", mesh.metadata.reference_element["Nxi"], mesh.global_state.vertices[mesh.global_state.connectivity, 0])
    J12_loc = np.einsum("ij,kj->ki", mesh.metadata.reference_element["Nxi"], mesh.global_state.vertices[mesh.global_state.connectivity, 1])
    J21_loc = np.einsum("ij,kj->ki", mesh.metadata.reference_element["Neta"], mesh.global_state.vertices[mesh.global_state.connectivity, 0])
    J22_loc = np.einsum("ij,kj->ki", mesh.metadata.reference_element["Neta"], mesh.global_state.vertices[mesh.global_state.connectivity, 1])
    detJ_loc = J11_loc * J22_loc - J12_loc * J21_loc

    mesh.derived_geometry.gauss_volumes = (
        2 * np.pi * mesh.metadata.reference_element["IPweights"][None, :] * detJ_loc[:, :] * mesh.derived_geometry.vertices_gauss[:, :, 0]
    )
    mesh.metadata.flags.gauss_volumes_initialized = True


def find_adjacent_elements(mesh, element_number):
    _ensure_full_mesh(mesh)
    vertices_numbers = mesh.global_state.connectivity[element_number]

    adjacent_numbers = []
    for number in vertices_numbers:
        idx = np.where(number == mesh.global_state.connectivity)
        for i in idx[0]:
            if i != element_number and i not in adjacent_numbers:
                adjacent_numbers.append(i)
    return adjacent_numbers
