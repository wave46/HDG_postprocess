import numpy as np


class SolutionTransport1D:
    def __init__(self, solution):
        self._solution = solution

    @property
    def available(self):
        return bool(self._solution.raw.transport_1d) and any(bool(item) for item in self._solution.raw.transport_1d)

    @property
    def datasets(self):
        if not self.available:
            return {}
        return dict(self._solution.raw.transport_1d[0])

    def get(self, name, default=None):
        if not self.available:
            return default
        value = self._solution.raw.transport_1d[0].get(name, default)
        if value is default:
            return default
        return np.asarray(value)

    @property
    def rho_grid(self):
        return self.get("rho_grid")

    @property
    def shell_weight(self):
        return self.get("shell_weight")

    @property
    def U_fs(self):
        return self.get("U_fs")

    @property
    def Q_fs(self):
        return self.get("Q_fs")

    @property
    def Q_rad_fs(self):
        return self.get("Q_rad_fs")

    @property
    def Q_rad_sum(self):
        return self.get("Q_rad_sum")

