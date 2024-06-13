import numpy as np
from .tools import softplus, double_softplus


def compute_iz_rate_NRL(te, te_min):

    Ery = 13.6
    te = np.maximum(te, te_min)
    E0 = te / Ery
    return 1e-11 * np.sqrt(E0) / (Ery**1.5) / (6 + E0) * np.exp(-1 / E0)


def compute_rec_rate_NRL(te, te_min):
    Ery = 13.6
    E0 = np.minimum(Ery / te, Ery / 0.1)
    sigmavrec = (
        5.2e-20 * np.sqrt(E0) * (0.43 + 0.5 * np.log(E0) + 0.469 * (E0 ** (-1.0 / 3)))
    )
    return sigmavrec


def eirene_fit_1D_log(te, params):
    """
    This routine calculates AMJUEL 1D rate (typically on temperature) in loglog space
    for given temperature and coefficients
    Also used for extrapolation of thermal OpenADAS case
    """
    result = np.zeros_like(te)
    for i in range(params.shape[0]):
        result += params[i] * np.log(te) ** i
    return result


def d_eirene_fit_1D_dte(te, params):
    """
    calculates derivative of AMJUEL 1D spline in loglog space
    """
    res = np.zeros_like(te)
    for i in range(1, params.shape[0]):
        res += i * params[i] * np.log(te) ** (i - 1)
    return res


def eirene_fit_1D(te, params, te_min):
    """
    This routine calculates extrapolated AMJUEL 1D rate (typically on temperature)
    for given temperature and coefficients
    Also used for extrapolation of thermal OpenADAS case
    """
    result = np.zeros_like(te)
    # good indices
    index_1 = te >= te_min
    result[index_1] = eirene_fit_1D_log(te[index_1], params)
    # bad indices
    index_2 = te < te_min
    result[index_2] = eirene_fit_1D_log(te_min * np.ones_like(te[index_2]), params)
    result[index_2] += d_eirene_fit_1D_dte(
        te_min * np.ones_like(te[index_2]), params
    ) * (np.log(te[index_2]) - np.log(te_min * np.ones_like(te[index_2])))
    return np.exp(result) / 1e6


def d_eirene_fit_dte(M, params):
    """
    partial derivative value in log log space
    """
    t_shape, ne_shape = params.shape
    te, ne = M
    result = 0
    for i in range(1, t_shape):
        for j in range(ne_shape):
            result += (
                params[i, j] * i * np.log(ne / 1e14) ** (j) * np.log(te) ** (i - 1)
            )
    return result


def eirene_fit(M, params, te_min, te_max, ne_min, ne_max):
    te, ne = M
    t_shape, ne_shape = params.shape
    result = np.zeros_like(te)
    ## region 1, in the range of applicability
    # part in good ne, te:
    index_1 = (ne >= ne_min) * (ne <= ne_max) * (te >= te_min) * (te <= te_max)
    for i in range(t_shape):
        for j in range(ne_shape):
            result[index_1] += (
                params[i, j]
                * np.log(ne[index_1] / 1e14) ** j
                * np.log(te[index_1]) ** i
            )
    ##region 2
    # ne<ne_min, te in good range:
    index_2 = (ne < ne_min) * (te >= te_min) * (te <= te_max)
    for i in range(t_shape):
        result[index_2] += params[i, 0] * np.log(te[index_2]) ** i

    ##region 3
    # ne<ne_min, te<te_min
    index_3 = (ne < ne_min) * (te < te_min)
    for i in range(t_shape):
        # for j in range(ne_shape):
        result[index_3] += params[i, 0] * np.log(te_min) ** i
    result[index_3] += d_eirene_fit_dte(
        np.vstack(
            [te_min * np.ones_like(te[index_3]), ne_min * np.ones_like(ne[index_3])]
        ),
        params,
    ) * (np.log(te[index_3]) - np.log(te_min))

    ##region 4
    # ne good, te<te_min
    index_4 = (ne >= ne_min) * (ne <= ne_max) * (te < te_min)
    for i in range(t_shape):
        for j in range(ne_shape):
            result[index_4] += (
                params[i, j] * np.log(ne[index_4] / 1e14) ** j * np.log(te_min) ** i
            )
    result[index_4] += d_eirene_fit_dte(
        np.vstack([te_min * np.ones_like(te[index_4]), ne[index_4]]), params
    ) * (np.log(te[index_4]) - np.log(te_min))

    ##region 5
    # ne>ne_max, te<te_min
    index_5 = (ne > ne_max) * (te < te_min)
    for i in range(t_shape):
        for j in range(ne_shape):
            result[index_5] += (
                params[i, j] * np.log(ne_max / 1e14) ** j * np.log(te_min) ** i
            )
    result[index_5] += d_eirene_fit_dte(
        np.vstack(
            [te_min * np.ones_like(te[index_5]), ne_max * np.ones_like(ne[index_5])]
        ),
        params,
    ) * (np.log(te[index_5]) - np.log(te_min))

    ##region 6
    # ne>ne_max, te norm
    index_6 = (ne > ne_max) * (te >= te_min) * (te <= te_max)
    for i in range(t_shape):
        for j in range(ne_shape):
            result[index_6] += (
                params[i, j] * np.log(ne_max / 1e14) ** j * np.log(te[index_6]) ** i
            )

    ##region 7
    # ne>ne_max, te>te_max
    index_7 = (ne > ne_max) * (te > te_max)
    for i in range(t_shape):
        for j in range(ne_shape):
            result[index_7] += (
                params[i, j] * np.log(ne_max / 1e14) ** j * np.log(te_max) ** i
            )
    result[index_7] += d_eirene_fit_dte(
        np.vstack(
            [te_max * np.ones_like(te[index_7]), ne_max * np.ones_like(ne[index_7])]
        ),
        params,
    ) * (np.log(te[index_7]) - np.log(te_max))

    ##region 8
    # ne normal, te>te_max
    index_8 = (ne >= ne_min) * (ne <= ne_max) * (te > te_max)
    for i in range(t_shape):
        for j in range(ne_shape):
            result[index_8] += (
                params[i, j] * np.log(ne[index_8] / 1e14) ** j * np.log(te_max) ** i
            )
    result[index_8] += d_eirene_fit_dte(
        np.vstack([te_max * np.ones_like(te[index_8]), ne[index_8]]), params
    ) * (np.log(te[index_8]) - np.log(te_max))

    ##region 9
    # ne<ne_min, te>te_max
    index_9 = (ne < ne_min) * (te > te_max)
    for i in range(t_shape):
        # for j in range(ne_shape):
        result[index_9] += params[i, 0] * np.log(te_max) ** i
    result[index_9] += d_eirene_fit_dte(
        np.vstack(
            [te_max * np.ones_like(te[index_9]), ne_min * np.ones_like(ne[index_9])]
        ),
        params,
    ) * (np.log(te[index_9]) - np.log(te_max))

    if (result > 1).any():
        print("WARNING")
        print(ne, te)
    result = np.exp(result) / 1e6
    return result


def calculate_iz_rate(te, ne, iz_parameters):
    """
    calculates iz rate for given te,ne
    """
    database = iz_parameters["database"]

    if database == "AMJUEL 2.1.5JH":
        alpha = iz_parameters["alpha"]
        te_min = iz_parameters["te_min"]
        te_max = iz_parameters["te_max"]
        ne_min = iz_parameters["ne_min"]
        ne_max = iz_parameters["ne_max"]
        return eirene_fit(np.vstack([te, ne]), alpha, te_min, te_max, ne_min, ne_max)
    elif database == "NRL":
        te_min = iz_parameters["te_min"]
        return compute_iz_rate_NRL(te, te_min)


def calculate_rec_rate(te, ne, parameters):
    """
    calculates rec rate for given te,ne
    """
    database = parameters["database"]

    if (database == "AMJUEL 2.1.8JH") or (database == "AMJUEL 2.1.8a"):
        alpha = parameters["alpha"]
        te_min = parameters["te_min"]
        te_max = parameters["te_max"]
        ne_min = parameters["ne_min"]
        ne_max = parameters["ne_max"]
        return eirene_fit(np.vstack([te, ne]), alpha, te_min, te_max, ne_min, ne_max)
    elif database == "NRL":
        te_min = parameters["te_min"]
        return compute_rec_rate_NRL(te, te_min)


def calculate_rec_rate_cons(solutions, parameters, T0, n0, Mref, tol=1e-20):
    """
    calculates iz rate for given te,ne
    """
    database = parameters["database"]

    dimensions = None
    if len(solutions.shape) > 2:
        dimensions = solutions.shape
        solutions = solutions.reshape(
            solutions.shape[0] * solutions.shape[1], solutions.shape[2]
        )
    if (database == "AMJUEL 2.1.8JH") or (database == "AMJUEL 2.1.8a"):
        alpha = parameters["alpha"]
        te_min = parameters["te_min"]
        te_max = parameters["te_max"]
        ne_min = parameters["ne_min"]
        ne_max = parameters["ne_max"]
        te = np.zeros_like(solutions[:, 0])
        ne = np.zeros_like(solutions[:, 0])
        # good U1 and U4
        good_idx = (solutions[:, 0].flatten() > tol) & (solutions[:, 3].flatten() > tol)

        te[good_idx] = (
            T0 * 2 / 3 / Mref * solutions[good_idx, 3] / solutions[good_idx, 0]
        )
        ne[good_idx] = n0 * solutions[good_idx, 0]
        te[~good_idx] = 1e-10
        ne[~good_idx] = n0 * 1e-20
        res = eirene_fit(np.vstack([te, ne]), alpha, te_min, te_max, ne_min, ne_max)

    elif database == "NRL":
        te_min = parameters["te_min"]
        te = np.zeros_like(solutions[:, 0])
        solutions_corrected = solutions.copy()
        solutions_corrected[:, 0] = np.maximum(solutions_corrected[:, 0], 1e-20)
        solutions_corrected[:, 3] = np.maximum(solutions_corrected[:, 3], 1e-20)
        te = T0 * 2 / 3 / Mref * solutions[:, 3] / solutions[:, 0]

        res = compute_rec_rate_NRL(te, te_min)

    if dimensions is not None:
        res = res.reshape(dimensions[0], dimensions[1])
    return res


def calculate_cx_rate(te, parameters):
    """
    calculates cx rate for given te,ne
    """
    database = parameters["database"]
    alpha = parameters["alpha"]
    te_min = parameters["te_min"]
    if database == "OpenADAS expanded":
        return eirene_fit_1D(te, alpha, te_min)


def calculate_cx_rate_cons(solutions, parameters, T0, Mref, tol=1e-20):
    """
    calculates cx rate for given te,ne
    """
    database = parameters["database"]
    alpha = parameters["alpha"]
    te_min = parameters["te_min"]
    dimensions = None
    if len(solutions.shape) > 2:
        dimensions = solutions.shape
        solutions = solutions.reshape(
            solutions.shape[0] * solutions.shape[1], solutions.shape[2]
        )

    if database == "OpenADAS expanded":
        te = np.zeros_like(solutions[:, 0])
        # good U1 and U4
        good_idx = (solutions[:, 0].flatten() > tol) & (solutions[:, 3].flatten() > tol)
        te[good_idx] = (
            T0 * 2 / 3 / Mref * solutions[good_idx, 3] / solutions[good_idx, 0]
        )
        te[~good_idx] = 1e-10

        res = eirene_fit_1D(te, alpha, te_min)
        if dimensions is not None:
            res = res.reshape(dimensions[0], dimensions[1])
        return res


def calculate_iz_rate_cons(solutions, iz_parameters, T0, n0, Mref, tol=1e-20):
    """
    calculates iz rate for given te,ne
    """
    database = iz_parameters["database"]
    alpha = iz_parameters["alpha"]
    te_min = iz_parameters["te_min"]
    te_max = iz_parameters["te_max"]
    ne_min = iz_parameters["ne_min"]
    ne_max = iz_parameters["ne_max"]
    dimensions = None
    if len(solutions.shape) > 2:
        dimensions = solutions.shape
        solutions = solutions.reshape(
            solutions.shape[0] * solutions.shape[1], solutions.shape[2]
        )
    if database == "AMJUEL 2.1.5JH":
        te = np.zeros_like(solutions[:, 0])
        ne = np.zeros_like(solutions[:, 0])
        # good U1 and U4
        good_idx = (solutions[:, 0].flatten() > tol) & (solutions[:, 3].flatten() > tol)

        te[good_idx] = (
            T0 * 2 / 3 / Mref * solutions[good_idx, 3] / solutions[good_idx, 0]
        )
        ne[good_idx] = n0 * solutions[good_idx, 0]
        te[~good_idx] = 1e-10
        ne[~good_idx] = n0 * 1e-20
        res = eirene_fit(np.vstack([te, ne]), alpha, te_min, te_max, ne_min, ne_max)
    elif database == "NRL":
        te = np.zeros_like(solutions[:, 0])
        te = T0 * 2 / 3 / Mref * solutions[:, 3] / solutions[:, 0]
        res = compute_iz_rate_NRL(te, te_min)
    if dimensions is not None:
        res = res.reshape(dimensions[0], dimensions[1])
    return res


def calculate_iz_source(te, ne, nn, iz_parameters):
    """
    calculates ionization source for given plasma density, electron temperature and neutral density
    """

    sigma_iz = calculate_iz_rate(te, ne, iz_parameters)
    return ne * nn * sigma_iz


def calculate_iz_source_cons(solutions, iz_parameters, T0, n0, Mref):
    """
    calculates ionization source for given conservative solutions
    todo make indexing not hardcoded
    """

    sigma_iz = calculate_iz_rate_cons(solutions, iz_parameters, T0, n0, Mref)
    if len(solutions.shape) > 2:
        res = n0**2 * solutions[:, :, 0] * solutions[:, :, -1] * sigma_iz
    else:
        res = n0**2 * solutions[:, 0] * solutions[:, -1] * sigma_iz
    return res


def calculate_cx_source(te, ne, nn, cx_parameters):
    """
    calculates charge-exchange source for given plasma density, electron temperature and neutral density
    """

    sigma_cx = calculate_cx_rate(te, cx_parameters)
    return ne * nn * sigma_cx


def calculate_cx_source_cons(solutions, cx_parameters, T0, n0, Mref):
    """
    calculates charge-exchange source for given conservative solutions
    todo make indexing not hardcoded
    """

    sigma_cx = calculate_cx_rate_cons(solutions, cx_parameters, T0, Mref)
    if len(solutions.shape) > 2:
        res = n0**2 * solutions[:, :, 0] * solutions[:, :, -1] * sigma_cx
    else:
        res = n0**2 * solutions[:, 0] * solutions[:, -1] * sigma_cx
    return res
