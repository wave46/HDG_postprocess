import numpy as np

from hdg_postprocess.routines.profiles import exponential_decay_fit, local_decay_length


def test_exponential_decay_fit_recovers_length():
    x = np.linspace(0.0, 0.1, 20)
    expected_lambda = 0.025
    y = 3.0 * np.exp(-x / expected_lambda)

    fit = exponential_decay_fit(x, y)

    assert np.isclose(fit["lambda"], expected_lambda)
    assert np.isclose(fit["amplitude"], 3.0)
    assert np.allclose(fit["fit"], y)


def test_local_decay_length_recovers_constant_length():
    x = np.linspace(0.0, 0.1, 20)
    expected_lambda = 0.025
    y = 3.0 * np.exp(-x / expected_lambda)

    local = local_decay_length(x, y)

    assert np.allclose(local, expected_lambda)
