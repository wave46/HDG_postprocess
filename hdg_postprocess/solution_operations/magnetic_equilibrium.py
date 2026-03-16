import numpy as np

from hdg_postprocess.routines.plasma import calculate_a, calculate_q_cyl


def define_magnetic_axis(solution):
    if not solution.combined_simple_solution:
        print("Comibining first simple solution full")
        solution.recombine_simple_full_solution()
    poloidal_flux_simple = solution.views.simple.equilibrium.poloidal_flux
    axis_r, axis_z = solution.mesh.vertices_glob[np.where(poloidal_flux_simple == poloidal_flux_simple.min())][0]
    solution._summary.equilibrium.axis.r = axis_r
    solution._summary.equilibrium.axis.z = axis_z


def define_minor_radii(solution, which="simple"):
    if which == "simple":
        if (solution.summary.equilibrium.axis.r is None) or (solution.summary.equilibrium.axis.z is None):
            solution.define_magnetic_axis()
        if not solution.mesh._combined_to_full:
            print("Comibining to full mesh")
            solution.mesh.recombine_full_mesh()
        solution.define_minor_radii(which="full")
        solution.views.simple.equilibrium.a = np.zeros(solution.mesh.vertices_glob.shape[0])
        solution.views.simple.equilibrium.a[solution.mesh.connectivity_glob.reshape(-1, 1).ravel()] = solution.views.glob.equilibrium.a.reshape(
            solution.views.glob.equilibrium.a.shape[0] * solution.views.glob.equilibrium.a.shape[1]
        )
    if which == "full":
        if (solution.summary.equilibrium.axis.r is None) or (solution.summary.equilibrium.axis.z is None):
            solution.define_magnetic_axis()
        if not solution.mesh._combined_to_full:
            print("Comibining to full mesh")
            solution.mesh.recombine_full_mesh()
        solution.views.glob.equilibrium.a = calculate_a(
            solution.mesh.vertices_glob[solution.mesh.connectivity_glob],
            solution.summary.equilibrium.axis.r,
            solution.summary.equilibrium.axis.z,
        )


def define_qcyl(solution, which="simple"):
    if which == "simple":
        if not solution.mesh._combined_to_full:
            print("Comibining to full mesh")
            solution.mesh.recombine_full_mesh()
        if not solution.combined_simple_solution:
            print("Comibining first simple solution full")
            solution.recombine_simple_full_solution()
        if solution.views.simple.equilibrium.a is None:
            solution.define_minor_radii()
        solution.define_qcyl(which="full")
        solution.views.simple.equilibrium.qcyl = np.zeros(solution.mesh.vertices_glob.shape[0])
        solution.views.simple.equilibrium.qcyl[solution.mesh.connectivity_glob.reshape(-1, 1).ravel()] = solution.views.glob.equilibrium.qcyl.reshape(
            solution.views.glob.equilibrium.qcyl.shape[0] * solution.views.glob.equilibrium.qcyl.shape[1]
        )
    elif which == "full":
        if not solution.mesh._combined_to_full:
            print("Comibining to full mesh")
            solution.mesh.recombine_full_mesh()
        if solution.views.glob.equilibrium.a is None:
            solution.define_minor_radii("full")
        glob_equilibrium = solution.views.glob.equilibrium
        solution.views.glob.equilibrium.qcyl = calculate_q_cyl(
            solution.mesh.vertices_glob[solution.mesh.connectivity_glob][:, :, 0],
            glob_equilibrium.magnetic_field[:, :, 0],
            glob_equilibrium.magnetic_field[:, :, 1],
            glob_equilibrium.magnetic_field[:, :, 2],
            glob_equilibrium.a,
        )
