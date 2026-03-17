from hdg_postprocess.core.solution import pointwise_fields as pointwise_fields_ops


class _PointwiseSection:
    def __init__(self, solution):
        self._solution = solution


class PointwisePlasma(_PointwiseSection):
    def n(self, r, z):
        return pointwise_fields_ops.n(self._solution, r, z)

    def ti(self, r, z):
        return pointwise_fields_ops.ti(self._solution, r, z)

    def te(self, r, z):
        return pointwise_fields_ops.te(self._solution, r, z)

    def u(self, r, z):
        return pointwise_fields_ops.u(self._solution, r, z)

    def cs(self, r, z):
        return pointwise_fields_ops.cs(self._solution, r, z)

    def mach(self, r, z):
        return pointwise_fields_ops.M(self._solution, r, z)

    def nn(self, r, z):
        return pointwise_fields_ops.nn(self._solution, r, z)

    def dnn(self, r, z):
        return pointwise_fields_ops.dnn(self._solution, r, z)

    def k(self, r, z):
        return pointwise_fields_ops.k(self._solution, r, z)

    def dk(self, r, z):
        return pointwise_fields_ops.dk(self._solution, r, z)

    def mfp_nn(self, r, z):
        return pointwise_fields_ops.mfp_nn(self._solution, r, z)

    def dynamic_pressure(self, r, z):
        return pointwise_fields_ops.p_dyn(self._solution, r, z)

    def ion_pressure(self, r, z):
        return pointwise_fields_ops.pi(self._solution, r, z)


class PointwiseGradients(_PointwiseSection):
    def ti(self, r, z, coordinate):
        return pointwise_fields_ops.grad_ti(self._solution, r, z, coordinate)

    def pi(self, r, z, coordinate):
        return pointwise_fields_ops.grad_pi(self._solution, r, z, coordinate)

    def ti_parallel(self, r, z):
        return pointwise_fields_ops.grad_ti_par(self._solution, r, z)

    def te(self, r, z, coordinate):
        return pointwise_fields_ops.grad_te(self._solution, r, z, coordinate)

    def te_parallel(self, r, z):
        return pointwise_fields_ops.grad_te_par(self._solution, r, z)


class PointwiseFluxes(_PointwiseSection):
    def particle_parallel(self, r, z):
        return pointwise_fields_ops.particle_flux_par(self._solution, r, z)

    def ion_heat_parallel_convective(self, r, z):
        return pointwise_fields_ops.ion_heat_flux_par_conv(self._solution, r, z)

    def ion_heat_parallel_conductive(self, r, z):
        return pointwise_fields_ops.ion_heat_flux_par_cond(self._solution, r, z)

    def ion_heat_parallel(self, r, z):
        return pointwise_fields_ops.ion_heat_flux_par(self._solution, r, z)

    def electron_heat_parallel_convective(self, r, z):
        return pointwise_fields_ops.electron_heat_flux_par_conv(self._solution, r, z)

    def electron_heat_parallel_conductive(self, r, z):
        return pointwise_fields_ops.electron_heat_flux_par_cond(self._solution, r, z)

    def electron_heat_parallel(self, r, z):
        return pointwise_fields_ops.electron_heat_flux_par(self._solution, r, z)


class PointwiseFields(_PointwiseSection):
    def psi(self, r, z):
        return pointwise_fields_ops.psi(self._solution, r, z)

    def magnetic_field(self, r, z, component):
        return pointwise_fields_ops.B(self._solution, r, z, component)

    def grad_magnetic_field(self, r, z, component, coordinate):
        return pointwise_fields_ops.grad_B(self._solution, r, z, component, coordinate)


class PointwiseSources(_PointwiseSection):
    def ionization_source(self, r, z):
        return pointwise_fields_ops.ionization_source_interp(self._solution, r, z)

    def ionization_rate(self, r, z):
        return pointwise_fields_ops.iz_rate(self._solution, r, z)

    def cx_rate(self, r, z):
        return pointwise_fields_ops.cx_rate(self._solution, r, z)

    def Q_e_loss_iz(self, r, z):
        return pointwise_fields_ops.Q_e_loss_iz(self._solution, r, z)

    def Q_e_loss_rec(self, r, z):
        return pointwise_fields_ops.Q_e_loss_rec(self._solution, r, z)

    def Q_e_gain_rec(self, r, z):
        return pointwise_fields_ops.Q_e_gain_rec(self._solution, r, z)

    def Q_e_loss_total(self, r, z):
        return pointwise_fields_ops.Q_e_loss_tot(self._solution, r, z)

    def Q_i_gain_iz(self, r, z):
        return pointwise_fields_ops.Q_i_gain_iz(self._solution, r, z)

    def Q_i_loss_rec(self, r, z):
        return pointwise_fields_ops.Q_i_loss_rec(self._solution, r, z)

    def Q_i_loss_cx(self, r, z):
        return pointwise_fields_ops.Q_i_loss_cx(self._solution, r, z)

    def Q_i_loss_total(self, r, z):
        return pointwise_fields_ops.Q_i_loss_tot(self._solution, r, z)

    def Q_loss_total(self, r, z):
        return pointwise_fields_ops.Q_loss_tot(self._solution, r, z)


class SolutionPointwise:
    def __init__(self, solution):
        self._solution = solution
        self.plasma = PointwisePlasma(solution)
        self.gradients = PointwiseGradients(solution)
        self.fluxes = PointwiseFluxes(solution)
        self.fields = PointwiseFields(solution)
        self.sources = PointwiseSources(solution)
