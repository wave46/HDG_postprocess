from hdg_postprocess.core.solution import neutrals as neutrals_ops
from hdg_postprocess.core.solution import neutral_flux_limiter as neutral_limiter_ops
from hdg_postprocess.core.solution import transport_postprocess as transport_postprocess_ops
from hdg_postprocess.core.solution import turbulent_model as turbulent_model_ops


class SolutionNeutrals:
    def __init__(self, solution):
        self._solution = solution

    def dnn(self, view="simple", with_nn_collision=False):
        if with_nn_collision:
            neutrals_ops.calculate_dnn_with_nn_collision(self._solution, view)
            if view == "full":
                return self._solution.views.glob.derived.dnn_with_nn_collision
            return self._solution.views.simple.derived.dnn_with_nn_collision
        neutrals_ops.calculate_dnn(self._solution, view)
        if view == "full":
            return self._solution.views.glob.derived.dnn
        return self._solution.views.simple.derived.dnn

    def mfp(self, view="simple"):
        neutrals_ops.calculate_mfp(self._solution, view)
        if view == "full":
            return self._solution.views.glob.derived.mfp
        return self._solution.views.simple.derived.mfp

    def limiter_diagnostic(self, name, view="element"):
        return neutral_limiter_ops.diagnostic_field(self._solution, name, view=view)

    def limiter_diagnostic_summary(self, activation_tol=1.0e-12):
        return neutral_limiter_ops.summarize_diagnostics(self._solution, activation_tol=activation_tol)

    def verify_limiter_diagnostics(self, *args, **kwargs):
        return neutral_limiter_ops.compare_neutral_flux_limiter_diagnostics(self._solution, *args, **kwargs)

    def recompute_limiter_diagnostics(self, *args, **kwargs):
        return neutral_limiter_ops.recompute_neutral_flux_limiter_diagnostics(self._solution, *args, **kwargs)

    @property
    def Dnn(self):
        return self.limiter_diagnostic("Dnn")

    @property
    def neutral_phi(self):
        return self.limiter_diagnostic("phi")

    @property
    def neutral_Deff(self):
        return self.limiter_diagnostic("D_eff")

    @property
    def neutral_gamma_unlim(self):
        return self.limiter_diagnostic("Gamma_unlim")

    @property
    def neutral_gamma_lim(self):
        return self.limiter_diagnostic("Gamma_lim")

    @property
    def neutral_gamma_max(self):
        return self.limiter_diagnostic("Gamma_max")

    @property
    def neutral_activation_ratio(self):
        return self.limiter_diagnostic("activation_ratio")


class SolutionTurbulence:
    def __init__(self, solution):
        self._solution = solution

    def dk(self, view="simple"):
        turbulent_model_ops.calculate_dk(self._solution, view)
        if view == "full":
            return self._solution.views.glob.derived.dk
        return self._solution.views.simple.derived.dk


class SolutionTransport:
    def __init__(self, solution):
        self._solution = solution

    def bohm(
        self,
        rho,
        *,
        rho_inner=0.8,
        rho_edge=0.99,
        method="gauss_shell",
        width=1e-3,
        derivative_mode="flux_normal",
        ion_mass_amu=2.0,
        ion_charge=1.0,
    ):
        return transport_postprocess_ops.bohm_profile(
            self._solution,
            rho,
            rho_inner=rho_inner,
            rho_edge=rho_edge,
            method=method,
            width=width,
            derivative_mode=derivative_mode,
            ion_mass_amu=ion_mass_amu,
            ion_charge=ion_charge,
        )

    def gyrobohm(
        self,
        rho,
        *,
        rho_inner=0.8,
        rho_edge=0.99,
        method="gauss_shell",
        width=1e-3,
        derivative_mode="flux_normal",
        ion_mass_amu=2.0,
        ion_charge=1.0,
    ):
        return transport_postprocess_ops.gyrobohm_profile(
            self._solution,
            rho,
            rho_inner=rho_inner,
            rho_edge=rho_edge,
            method=method,
            width=width,
            derivative_mode=derivative_mode,
            ion_mass_amu=ion_mass_amu,
            ion_charge=ion_charge,
        )

    def bohm_gyrobohm(
        self,
        rho,
        *,
        rho_inner=0.8,
        rho_edge=0.99,
        method="gauss_shell",
        width=1e-3,
        derivative_mode="flux_normal",
        ion_mass_amu=2.0,
        ion_charge=1.0,
    ):
        return transport_postprocess_ops.mixed_bohm_gyrobohm(
            self._solution,
            rho,
            rho_inner=rho_inner,
            rho_edge=rho_edge,
            method=method,
            width=width,
            derivative_mode=derivative_mode,
            ion_mass_amu=ion_mass_amu,
            ion_charge=ion_charge,
        )
