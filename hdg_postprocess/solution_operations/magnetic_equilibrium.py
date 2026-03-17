import numpy as np

from hdg_postprocess.solution_operations import preparation as prep_ops
from hdg_postprocess.routines.plasma import calculate_a, calculate_q_cyl


def define_magnetic_axis(solution):
    prep_ops.ensure_simple_solution(solution)
    poloidal_flux_simple = solution.views.simple.equilibrium.poloidal_flux
    axis_r, axis_z = solution.mesh.global_state.vertices[np.where(poloidal_flux_simple == poloidal_flux_simple.min())][0]
    solution._summary.equilibrium.axis.r = axis_r
    solution._summary.equilibrium.axis.z = axis_z


def define_minor_radii(solution, which="simple"):
    if which == "simple":
        if (solution.summary.equilibrium.axis.r is None) or (solution.summary.equilibrium.axis.z is None):
            define_magnetic_axis(solution)
        if not solution.mesh.metadata.flags.combined_to_full:
            print("Combining full mesh first")
            solution.mesh.assembly.full()
        define_minor_radii(solution, which="full")
        solution.views.simple.equilibrium.a = prep_ops.project_full_to_simple(solution, solution.views.glob.equilibrium.a)
    if which == "full":
        if (solution.summary.equilibrium.axis.r is None) or (solution.summary.equilibrium.axis.z is None):
            define_magnetic_axis(solution)
        if not solution.mesh.metadata.flags.combined_to_full:
            print("Combining full mesh first")
            solution.mesh.assembly.full()
        solution.views.glob.equilibrium.a = calculate_a(
            solution.mesh.global_state.vertices[solution.mesh.global_state.connectivity],
            solution.summary.equilibrium.axis.r,
            solution.summary.equilibrium.axis.z,
        )


def define_qcyl(solution, which="simple"):
    if which == "simple":
        if not solution.mesh.metadata.flags.combined_to_full:
            print("Combining full mesh first")
            solution.mesh.assembly.full()
        prep_ops.ensure_simple_solution(solution)
        if solution.views.simple.equilibrium.a is None:
            define_minor_radii(solution)
        define_qcyl(solution, which="full")
        solution.views.simple.equilibrium.qcyl = prep_ops.project_full_to_simple(
            solution, solution.views.glob.equilibrium.qcyl
        )
    elif which == "full":
        if not solution.mesh.metadata.flags.combined_to_full:
            print("Combining full mesh first")
            solution.mesh.assembly.full()
        if solution.views.glob.equilibrium.a is None:
            define_minor_radii(solution, "full")
        glob_equilibrium = solution.views.glob.equilibrium
        solution.views.glob.equilibrium.qcyl = calculate_q_cyl(
            solution.mesh.global_state.vertices[solution.mesh.global_state.connectivity][:, :, 0],
            glob_equilibrium.magnetic_field[:, :, 0],
            glob_equilibrium.magnetic_field[:, :, 1],
            glob_equilibrium.magnetic_field[:, :, 2],
            glob_equilibrium.a,
        )
