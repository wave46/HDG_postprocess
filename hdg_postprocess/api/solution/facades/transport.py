from hdg_postprocess.core.solution import neutrals as neutrals_ops
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
