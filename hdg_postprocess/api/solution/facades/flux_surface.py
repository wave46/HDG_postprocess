from hdg_postprocess.core.solution import surfaces as surface_ops


class SolutionFluxSurface:
    def __init__(self, solution):
        self._solution = solution

    def average(self, field, rho, *, method="gauss_shell", width=1e-3):
        return surface_ops.average_on_surfaces(self._solution, field, rho, method=method, width=width)

    def te(self, rho, *, method="gauss_shell", width=1e-3):
        return surface_ops.te_on_surfaces(self._solution, rho, method=method, width=width)

    def delta_te(self, *, rho_inner=0.8, rho_outer=1.0, method="gauss_shell", width=1e-3):
        return surface_ops.delta_te_on_surfaces(
            self._solution,
            rho_inner=rho_inner,
            rho_outer=rho_outer,
            method=method,
            width=width,
        )

    def minor_radius(self, rho, *, method="gauss_shell", width=1e-3):
        return surface_ops.minor_radius_on_surfaces(self._solution, rho, method=method, width=width)

    def major_radius(self, rho, *, method="gauss_shell", width=1e-3):
        return surface_ops.major_radius_on_surfaces(self._solution, rho, method=method, width=width)

    def epsilon(self, rho, *, method="gauss_shell", width=1e-3):
        return surface_ops.epsilon_on_surfaces(self._solution, rho, method=method, width=width)

    def q(self, rho, *, method="gauss_shell", width=1e-3):
        return surface_ops.q_on_surfaces(self._solution, rho, method=method, width=width)

    def collisionality(
        self,
        rho,
        *,
        z_effective=1.0,
        coulomb_logarithm=None,
        method="gauss_shell",
        width=1e-3,
    ):
        return surface_ops.collisionality_on_surfaces(
            self._solution,
            rho,
            z_effective=z_effective,
            coulomb_logarithm=coulomb_logarithm,
            method=method,
            width=width,
        )

    def pinch_factor(
        self,
        rho,
        *,
        z_effective=1.0,
        coulomb_logarithm=None,
        threshold=0.04,
        method="gauss_shell",
        width=1e-3,
    ):
        return surface_ops.pinch_factor_on_surfaces(
            self._solution,
            rho,
            z_effective=z_effective,
            coulomb_logarithm=coulomb_logarithm,
            threshold=threshold,
            method=method,
            width=width,
        )


    def rho(self, *, target="node"):
        return surface_ops.rho_field(self._solution, target=target)

    def project(self, rho, values, *, target="node"):
        return surface_ops.project_profile_to_solution(self._solution, rho, values, target=target)
    def pinch_velocity(
        self,
        rho,
        diffusivity,
        *,
        rho_edge=0.99,
        z_effective=1.0,
        coulomb_logarithm=None,
        threshold=0.04,
        method="gauss_shell",
        width=1e-3,
    ):
        return surface_ops.pinch_velocity_on_surfaces(
            self._solution,
            rho,
            diffusivity,
            rho_edge=rho_edge,
            z_effective=z_effective,
            coulomb_logarithm=coulomb_logarithm,
            threshold=threshold,
            method=method,
            width=width,
        )
