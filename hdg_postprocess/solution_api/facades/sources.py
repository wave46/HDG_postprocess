from hdg_postprocess.solution_operations import plasma_sources as plasma_source_ops


class SourceRates:
    def __init__(self, sources):
        self._sources = sources

    def ionization(self, view="simple"):
        return self._sources._run_rate(plasma_source_ops.calculate_ionization_rate, "ionization_simple", view)

    def recombination(self, view="simple"):
        return self._sources._run_rate(plasma_source_ops.calculate_recombination_rate, "recombination_simple", view)

    def cx(self, view="simple"):
        return self._sources._run_rate(plasma_source_ops.calculate_cx_rate, "cx_simple", view)


class SolutionSources:
    def __init__(self, solution):
        self._solution = solution
        self.rates = SourceRates(self)

    def _target_view(self, view):
        return self._solution.views.glob if view == "full" else getattr(self._solution.views, view)

    def _run_view_field(self, calculator, field_name, view):
        calculator(self._solution, view)
        return getattr(self._target_view(view).sources, field_name)

    def _run_rate(self, calculator, rate_name, view):
        calculator(self._solution, view)
        return getattr(self._solution.atomic_rates, rate_name)

    def ohmic(self, view="simple"):
        return self._run_view_field(plasma_source_ops.calculate_ohmic_source, "ohmic_source", view)

    def ionization_rate(self, view="simple"):
        return self.rates.ionization(view)

    def recombination_rate(self, view="simple"):
        return self.rates.recombination(view)

    def cx_rate(self, view="simple"):
        return self.rates.cx(view)

    def ionization(self, view="simple"):
        return self._run_view_field(plasma_source_ops.calculate_ionization_source, "ionization_source", view)

    def ion_gain_iz(self, view="simple"):
        return self._run_view_field(plasma_source_ops.calculate_ion_gain_due_to_iz, "ion_gain_iz", view)

    def electron_sink_iz(self, view="simple"):
        return self._run_view_field(plasma_source_ops.calculate_electron_sink_due_to_iz, "electron_sink_iz", view)

    def electron_sink_rec(self, view="simple"):
        return self._run_view_field(plasma_source_ops.calculate_electron_sink_due_to_rec, "electron_sink_rec", view)

    def electron_gain_rec(self, view="simple"):
        return self._run_view_field(plasma_source_ops.calculate_electron_gain_due_to_rec, "electron_gain_rec", view)

    def ion_sink_rec(self, view="simple"):
        return self._run_view_field(plasma_source_ops.calculate_ion_sink_due_to_rec, "ion_sink_rec", view)

    def ion_sink_cx(self, view="simple"):
        return self._run_view_field(plasma_source_ops.calculate_ion_sink_due_to_cx, "ion_sink_cx", view)

    def electron_sink_cooling_factor(self, view="simple"):
        return self._run_view_field(
            plasma_source_ops.calculate_electron_sink_due_to_cooling_factor,
            "electron_sink_cooling_factor",
            view,
        )

    def cooling_factor(self, view="simple"):
        return self._run_view_field(plasma_source_ops.calculate_cooling_factor, "cooling_factor", view)

    def cx(self, view="simple"):
        return self._run_view_field(plasma_source_ops.calculate_cx_source, "cx_source", view)
