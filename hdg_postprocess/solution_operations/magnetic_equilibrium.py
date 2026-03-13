import numpy as np

from hdg_postprocess.routines.plasma import calculate_a, calculate_q_cyl


def define_magnetic_axis(solution):
    if not solution.combined_simple_solution:
        print("Comibining first simple solution full")
        solution.recombine_simple_full_solution()
    solution._r_axis, solution._z_axis = solution.mesh.vertices_glob[
        np.where(solution.poloidal_flux_simple == solution.poloidal_flux_simple.min())
    ][0]


def define_minor_radii(solution, which="simple"):
    if which == "simple":
        if (solution._r_axis is None) or (solution._z_axis is None):
            solution.define_magnetic_axis()
        if not solution.mesh._combined_to_full:
            print("Comibining to full mesh")
            solution.mesh.recombine_full_mesh()
        solution.define_minor_radii(which="full")
        solution._a_simple = np.zeros(solution.mesh.vertices_glob.shape[0])
        solution._a_simple[solution.mesh.connectivity_glob.reshape(-1, 1).ravel()] = solution._a_glob.reshape(
            solution._a_glob.shape[0] * solution._a_glob.shape[1]
        )
    if which == "full":
        if (solution._r_axis is None) or (solution._z_axis is None):
            solution.define_magnetic_axis()
        if not solution.mesh._combined_to_full:
            print("Comibining to full mesh")
            solution.mesh.recombine_full_mesh()
        solution._a_glob = calculate_a(
            solution.mesh.vertices_glob[solution.mesh.connectivity_glob], solution.r_axis, solution.z_axis
        )


def define_qcyl(solution, which="simple"):
    if which == "simple":
        if not solution.mesh._combined_to_full:
            print("Comibining to full mesh")
            solution.mesh.recombine_full_mesh()
        if not solution.combined_simple_solution:
            print("Comibining first simple solution full")
            solution.recombine_simple_full_solution()
        if solution.a_simple is None:
            solution.define_minor_radii()
        solution.define_qcyl(which="full")
        solution._qcyl_simple = np.zeros(solution.mesh.vertices_glob.shape[0])
        solution._qcyl_simple[solution.mesh.connectivity_glob.reshape(-1, 1).ravel()] = solution._qcyl_glob.reshape(
            solution._qcyl_glob.shape[0] * solution._qcyl_glob.shape[1]
        )
    elif which == "full":
        if not solution.mesh._combined_to_full:
            print("Comibining to full mesh")
            solution.mesh.recombine_full_mesh()
        if solution.a_glob is None:
            solution.define_minor_radii("full")
        solution._qcyl_glob = calculate_q_cyl(
            solution.mesh.vertices_glob[solution.mesh.connectivity_glob][:, :, 0],
            solution.magnetic_field_glob[:, :, 0],
            solution.magnetic_field_glob[:, :, 1],
            solution.magnetic_field_glob[:, :, 2],
            solution.a_glob,
        )
