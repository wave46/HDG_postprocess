import numpy as np

from hdg_postprocess.routines.plasma import calculate_dk_cons


def calculate_dk(solution, which="simple"):
    """
    Calculate turbulent diffusion derived from the k-equation model.
    """
    if which == "simple":
        if solution.dk_parameters is None:
            raise ValueError("Please, provide turbulent diffusion settings for the simulation")

        if not solution._combined_simple_solution:
            print("Initializing physical solution first")
            solution.recombine_simple_full_solution()
        if (solution.r_axis is None) or (solution.z_axis is None):
            solution.define_magnetic_axis()
        if solution.a_simple is None:
            solution.define_minor_radii(which="simple")
        if solution.qcyl_simple is None:
            solution.define_qcyl(which="simple")
        solution.calculate_dk(which="full")
        solution._dk_simple = np.zeros(solution.mesh.vertices_glob.shape[0])
        solution._dk_simple[solution.mesh.connectivity_glob.reshape(-1, 1).ravel()] = solution._dk_glob.reshape(
            solution._dk_glob.shape[0] * solution._dk_glob.shape[1]
        )

    if which == "full":
        if solution.dk_parameters is None:
            raise ValueError("Please, provide turbulent diffusion settings for the simulation")

        if not solution._combined_simple_solution:
            print("Initializing physical solution first")
            solution.recombine_simple_full_solution()
        if (solution.r_axis is None) or (solution.z_axis is None):
            solution.define_magnetic_axis()
        if solution.a_glob is None:
            solution.define_minor_radii(which="full")
        if solution.qcyl_glob is None:
            solution.define_qcyl(which="full")

        solution._dk_glob = calculate_dk_cons(
            solution.solution_glob,
            solution.dk_parameters,
            solution.qcyl_glob,
            solution.mesh.vertices_glob[solution.mesh.connectivity_glob][:, :, 0]
            / solution.parameters["adimensionalization"]["length_scale"],
            solution.parameters["adimensionalization"]["length_scale"] ** 2
            / solution.parameters["adimensionalization"]["time_scale"],
            solution.cons_idx,
        )
