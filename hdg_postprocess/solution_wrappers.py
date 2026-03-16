from hdg_postprocess.solution_operations import analysis as analysis_ops
from hdg_postprocess.solution_operations import assembly as assembly_ops
from hdg_postprocess.solution_operations import boundary as boundary_ops
from hdg_postprocess.solution_operations import magnetic_equilibrium as equilibrium_ops
from hdg_postprocess.solution_operations import neutrals as neutrals_ops
from hdg_postprocess.solution_operations import physical as physical_ops
from hdg_postprocess.solution_operations import plasma_sources as plasma_source_ops
from hdg_postprocess.solution_operations import plotting as plotting_ops
from hdg_postprocess.solution_operations import pointwise_fields as pointwise_fields_ops
from hdg_postprocess.solution_operations import sampling as sampling_ops
from hdg_postprocess.solution_operations import turbulent_model as turbulent_model_ops


def _make_delegate(name, func, doc=None):
    def method(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    method.__name__ = name
    method.__qualname__ = f"HDGsolution.{name}"
    method.__doc__ = doc
    return method


_WRAPPER_SPECS = [
    ("recombine_full_solution", assembly_ops.recombine_full_solution, "Recombine raw solutions into one single mesh."),
    (
        "recombine_simple_full_solution",
        assembly_ops.recombine_simple_full_solution,
        "Create the vertex-only convenience view used for lightweight plotting and sampling.",
    ),
    ("recombine_boundary_solution", assembly_ops.recombine_boundary_solution, "Extract boundary solution data."),
    ("calculate_in_gauss_points", assembly_ops.calculate_in_gauss_points, "Evaluate solution data in volume Gauss points."),
    (
        "calculate_in_boundary_gauss_points",
        assembly_ops.calculate_in_boundary_gauss_points,
        "Evaluate solution data in boundary Gauss points.",
    ),
    ("summary_along_the_wall", boundary_ops.summary_along_the_wall, "Summarize physical quantities along the wall."),
    ("plot_overview", plotting_ops.plot_overview, "Plot conservative overview fields."),
    ("plot_overview_difference", plotting_ops.plot_overview_difference, "Plot conservative differences between solutions."),
    ("init_phys_variables", physical_ops.init_phys_variables, "Initialize physical variables on the requested view."),
    ("cons2phys", physical_ops.cons2phys, "Convert conservative data to physical variables."),
    ("plot_overview_physical", plotting_ops.plot_overview_physical, "Plot physical overview fields."),
    (
        "plot_overview_physical_difference",
        plotting_ops.plot_overview_physical_difference,
        "Plot physical differences between solutions.",
    ),
    ("plot_variables_overview", plotting_ops.plot_variables_overview, "Plot a custom overview selection."),
    ("define_magnetic_axis", equilibrium_ops.define_magnetic_axis, "Define magnetic axis from the equilibrium field."),
    ("define_minor_radii", equilibrium_ops.define_minor_radii, "Define minor radii on the requested view."),
    ("define_qcyl", equilibrium_ops.define_qcyl, "Define cylindrical safety factor on the requested view."),
    ("calculate_variables_along_line", sampling_ops.calculate_variables_along_line, "Sample variables along a line."),
    ("save_summary_line", sampling_ops.save_summary_line, "Save a sampled line summary to disk."),
    ("calculate_ohmic_source", plasma_source_ops.calculate_ohmic_source, "Compute ohmic source on the requested view."),
    ("calculate_power_balance", analysis_ops.calculate_power_balance, "Compute the power-balance summary."),
    ("calculate_volumetric_sources", analysis_ops.calculate_volumetric_sources, "Compute volumetric source totals."),
    ("calculate_power_losses_to_wall", analysis_ops.calculate_power_losses_to_wall, "Compute wall-loss summaries."),
    ("calculate_boundary_summary", boundary_ops.calculate_boundary_summary, "Compute structured boundary summaries."),
    ("calculate_ionization_rate", plasma_source_ops.calculate_ionization_rate, "Compute ionization rate coefficients."),
    (
        "calculate_recombination_rate",
        plasma_source_ops.calculate_recombination_rate,
        "Compute recombination rate coefficients.",
    ),
    ("calculate_cx_rate", plasma_source_ops.calculate_cx_rate, "Compute charge-exchange rate coefficients."),
    ("calculate_dnn", neutrals_ops.calculate_dnn, "Compute neutral diffusion."),
    (
        "calculate_dnn_with_nn_collision",
        neutrals_ops.calculate_dnn_with_nn_collision,
        "Compute neutral diffusion including neutral-neutral collisions.",
    ),
    ("calculate_dk", turbulent_model_ops.calculate_dk, "Compute plasma diffusion from the turbulent model."),
    ("calculate_mfp", neutrals_ops.calculate_mfp, "Compute neutral mean-free path."),
    ("calculate_ionization_source", plasma_source_ops.calculate_ionization_source, "Compute ionization source."),
    ("calculate_ion_gain_due_to_iz", plasma_source_ops.calculate_ion_gain_due_to_iz, "Compute ion ionization gain."),
    ("calculate_ion_sink_due_to_rec", plasma_source_ops.calculate_ion_sink_due_to_rec, "Compute ion recombination loss."),
    ("calculate_ion_sink_due_to_cx", plasma_source_ops.calculate_ion_sink_due_to_cx, "Compute ion charge-exchange loss."),
    (
        "calculate_electron_sink_due_to_iz",
        plasma_source_ops.calculate_electron_sink_due_to_iz,
        "Compute electron ionization loss.",
    ),
    (
        "calculate_electron_sink_due_to_rec",
        plasma_source_ops.calculate_electron_sink_due_to_rec,
        "Compute electron recombination loss.",
    ),
    (
        "calculate_electron_gain_due_to_rec",
        plasma_source_ops.calculate_electron_gain_due_to_rec,
        "Compute electron recombination gain.",
    ),
    (
        "calculate_electron_sink_due_to_cooling_factor",
        plasma_source_ops.calculate_electron_sink_due_to_cooling_factor,
        "Compute electron cooling-factor loss.",
    ),
    ("calculate_cooling_factor", plasma_source_ops.calculate_cooling_factor, "Compute cooling-factor field."),
    ("calculate_cx_source", plasma_source_ops.calculate_cx_source, "Compute charge-exchange source."),
    ("define_interpolators", sampling_ops.define_interpolators, "Define solution interpolators."),
    ("n", pointwise_fields_ops.n, "Sample density at a point."),
    ("ti", pointwise_fields_ops.ti, "Sample ion temperature at a point."),
    ("te", pointwise_fields_ops.te, "Sample electron temperature at a point."),
    ("u", pointwise_fields_ops.u, "Sample parallel velocity at a point."),
    ("cs", pointwise_fields_ops.cs, "Sample sound speed at a point."),
    ("M", pointwise_fields_ops.M, "Sample Mach number at a point."),
    ("nn", pointwise_fields_ops.nn, "Sample neutral density at a point."),
    ("ionization_source_interp", pointwise_fields_ops.ionization_source_interp, "Sample ionization source at a point."),
    ("iz_rate", pointwise_fields_ops.iz_rate, "Sample ionization rate at a point."),
    ("cx_rate", pointwise_fields_ops.cx_rate, "Sample charge-exchange rate at a point."),
    ("dnn", pointwise_fields_ops.dnn, "Sample neutral diffusion at a point."),
    ("k", pointwise_fields_ops.k, "Sample turbulent energy at a point."),
    ("dk", pointwise_fields_ops.dk, "Sample plasma diffusion at a point."),
    ("mfp_nn", pointwise_fields_ops.mfp_nn, "Sample neutral mean-free path at a point."),
    ("p_dyn", pointwise_fields_ops.p_dyn, "Sample dynamic pressure at a point."),
    ("pi", pointwise_fields_ops.pi, "Sample ion pressure at a point."),
    ("grad_ti", pointwise_fields_ops.grad_ti, "Sample ion-temperature gradient at a point."),
    ("grad_pi", pointwise_fields_ops.grad_pi, "Sample ion-pressure gradient at a point."),
    ("grad_ti_par", pointwise_fields_ops.grad_ti_par, "Sample parallel ion-temperature gradient at a point."),
    ("grad_te", pointwise_fields_ops.grad_te, "Sample electron-temperature gradient at a point."),
    ("grad_te_par", pointwise_fields_ops.grad_te_par, "Sample parallel electron-temperature gradient at a point."),
    ("particle_flux_par", pointwise_fields_ops.particle_flux_par, "Sample parallel particle flux at a point."),
    (
        "ion_heat_flux_par_conv",
        pointwise_fields_ops.ion_heat_flux_par_conv,
        "Sample convective ion heat flux at a point.",
    ),
    (
        "ion_heat_flux_par_cond",
        pointwise_fields_ops.ion_heat_flux_par_cond,
        "Sample conductive ion heat flux at a point.",
    ),
    ("ion_heat_flux_par", pointwise_fields_ops.ion_heat_flux_par, "Sample total ion heat flux at a point."),
    (
        "electron_heat_flux_par_conv",
        pointwise_fields_ops.electron_heat_flux_par_conv,
        "Sample convective electron heat flux at a point.",
    ),
    (
        "electron_heat_flux_par_cond",
        pointwise_fields_ops.electron_heat_flux_par_cond,
        "Sample conductive electron heat flux at a point.",
    ),
    ("electron_heat_flux_par", pointwise_fields_ops.electron_heat_flux_par, "Sample total electron heat flux at a point."),
    ("psi", pointwise_fields_ops.psi, "Sample poloidal flux at a point."),
    ("B", pointwise_fields_ops.B, "Sample magnetic field at a point."),
    ("grad_B", pointwise_fields_ops.grad_B, "Sample magnetic-field gradient at a point."),
    ("Q_e_loss_iz", pointwise_fields_ops.Q_e_loss_iz, "Sample electron ionization loss at a point."),
    ("Q_e_loss_rec", pointwise_fields_ops.Q_e_loss_rec, "Sample electron recombination loss at a point."),
    ("Q_e_gain_rec", pointwise_fields_ops.Q_e_gain_rec, "Sample electron recombination gain at a point."),
    ("Q_e_loss_tot", pointwise_fields_ops.Q_e_loss_tot, "Sample total electron loss at a point."),
    ("Q_i_gain_iz", pointwise_fields_ops.Q_i_gain_iz, "Sample ion ionization gain at a point."),
    ("Q_i_loss_rec", pointwise_fields_ops.Q_i_loss_rec, "Sample ion recombination loss at a point."),
    ("Q_i_loss_cx", pointwise_fields_ops.Q_i_loss_cx, "Sample ion charge-exchange loss at a point."),
    ("Q_i_loss_tot", pointwise_fields_ops.Q_i_loss_tot, "Sample total ion loss at a point."),
    ("Q_loss_tot", pointwise_fields_ops.Q_loss_tot, "Sample total power loss at a point."),
]


def attach_solution_wrappers(cls):
    for name, func, doc in _WRAPPER_SPECS:
        setattr(cls, name, _make_delegate(name, func, doc=doc))

