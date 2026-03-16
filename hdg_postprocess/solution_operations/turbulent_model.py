import numpy as np

from hdg_postprocess.routines.plasma import calculate_dk_cons


def calculate_dk(solution, which="simple"):
    """
    Calculate turbulent diffusion derived from the k-equation model.
    """
    if which == "simple":
        if solution.additional_parameters.turbulence is None:
            raise ValueError("Please, provide turbulent diffusion settings for the simulation")

        if not solution.metadata.flags.combined_simple_solution:
            print("Initializing physical solution first")
            solution.assembly.simple()
        if (solution.summary.equilibrium.axis.r is None) or (solution.summary.equilibrium.axis.z is None):
            solution.equilibrium.define_axis()
        if solution.views.simple.equilibrium.a is None:
            solution.equilibrium.define_minor_radii(view="simple")
        if solution.views.simple.equilibrium.qcyl is None:
            solution.equilibrium.define_qcyl(view="simple")
        calculate_dk(solution, which="full")
        solution.views.simple.derived.dk = np.zeros(solution.mesh.global_state.vertices.shape[0])
        solution.views.simple.derived.dk[solution.mesh.global_state.connectivity.reshape(-1, 1).ravel()] = solution.views.glob.derived.dk.reshape(
            solution.views.glob.derived.dk.shape[0] * solution.views.glob.derived.dk.shape[1]
        )

    if which == "full":
        if solution.additional_parameters.turbulence is None:
            raise ValueError("Please, provide turbulent diffusion settings for the simulation")

        if not solution.metadata.flags.combined_simple_solution:
            print("Initializing physical solution first")
            solution.assembly.simple()
        if (solution.summary.equilibrium.axis.r is None) or (solution.summary.equilibrium.axis.z is None):
            solution.equilibrium.define_axis()
        if solution.views.glob.equilibrium.a is None:
            solution.equilibrium.define_minor_radii(view="glob")
        if solution.views.glob.equilibrium.qcyl is None:
            solution.equilibrium.define_qcyl(view="glob")

        solution.views.glob.derived.dk = calculate_dk_cons(
            solution.views.glob.solution.conservative,
            solution.additional_parameters.turbulence,
            solution.views.glob.equilibrium.qcyl,
            solution.mesh.global_state.vertices[solution.mesh.global_state.connectivity][:, :, 0]
            / solution.parameters["adimensionalization"]["length_scale"],
            solution.parameters["adimensionalization"]["length_scale"] ** 2
            / solution.parameters["adimensionalization"]["time_scale"],
            solution.metadata.indices.conservative,
        )
