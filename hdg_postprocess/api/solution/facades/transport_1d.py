import numpy as np


class SolutionTransport1DGroup:
    def __init__(self, data):
        self._data = data or {}

    def __bool__(self):
        return bool(self._data)

    def __contains__(self, name):
        return name in self._data

    def __getitem__(self, name):
        return np.asarray(self._data[name])

    def __getattr__(self, name):
        if name in self._data:
            return np.asarray(self._data[name])
        raise AttributeError(name)

    def get(self, name, default=None):
        value = self._data.get(name, default)
        if value is default:
            return default
        return np.asarray(value)

    def keys(self):
        return self._data.keys()

    def items(self):
        return ((key, np.asarray(value)) for key, value in self._data.items())

    def as_dict(self):
        return {key: np.asarray(value) for key, value in self._data.items()}


class SolutionTransport1D:
    def __init__(self, solution):
        self._solution = solution

    def _root(self):
        if not self._solution.raw.transport_1d:
            return {}
        return self._solution.raw.transport_1d[0]

    @property
    def available(self):
        root = self._root()
        return bool(root)

    @property
    def datasets(self):
        if not self.available:
            return {}
        return dict(self._root())

    def _group(self, name):
        if not self.available:
            return SolutionTransport1DGroup({})
        group = self._root().get(name, {})
        return SolutionTransport1DGroup(group)

    @property
    def profiles(self):
        return self._group("profiles")

    @property
    def coefficients(self):
        return self._group("coefficients")

    @property
    def params(self):
        return self._group("params")

    def get(self, name, default=None):
        if not self.available:
            return default
        root = self._root()
        if name in root:
            value = root.get(name, default)
        elif name in root.get("profiles", {}):
            value = root["profiles"].get(name, default)
        elif name in root.get("coefficients", {}):
            value = root["coefficients"].get(name, default)
        elif name in root.get("params", {}):
            value = root["params"].get(name, default)
        else:
            value = default
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

    @property
    def chi_i_fs(self):
        return self.get("chi_i_fs")

    @property
    def chi_e_fs(self):
        return self.get("chi_e_fs")

    @property
    def d_fs(self):
        return self.get("d_fs")

    @property
    def nu_mom_fs(self):
        return self.get("nu_mom_fs")

    @property
    def vpinch_fs(self):
        return self.get("vpinch_fs")
