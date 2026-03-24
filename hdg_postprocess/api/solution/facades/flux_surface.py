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
