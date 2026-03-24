from hdg_postprocess.core.solution import neutrals as neutrals_ops
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
