from hdg_postprocess.core.solution import preparation as prep_ops
from hdg_postprocess.routines.plasma import calculate_dk_cons


def calculate_dk(solution, which="simple"):
    """
    Calculate turbulent diffusion derived from the k-equation model.
    """
    _require_turbulence_settings(solution)
    if which == "simple":
        prep_ops.ensure_simple_solution(solution)
        _ensure_turbulence_geometry(solution, view="simple")
        calculate_dk(solution, which="full")
        solution.views.simple.derived.dk = prep_ops.project_full_to_simple(solution, solution.views.glob.derived.dk)

    elif which == "full":
        prep_ops.ensure_simple_solution(solution)
        _ensure_turbulence_geometry(solution, view="glob")

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


def _require_turbulence_settings(solution):
    if solution.additional_parameters.turbulence is None:
        raise ValueError("Please, provide turbulent diffusion settings for the simulation")


def _ensure_turbulence_geometry(solution, view):
    if (solution.summary.equilibrium.axis.r is None) or (solution.summary.equilibrium.axis.z is None):
        solution.equilibrium.define_axis()

    equilibrium_view = solution.views.glob.equilibrium if view == "glob" else solution.views.simple.equilibrium
    if equilibrium_view.a is None:
        solution.equilibrium.define_minor_radii(view=view)
    if equilibrium_view.qcyl is None:
        solution.equilibrium.define_qcyl(view=view)
