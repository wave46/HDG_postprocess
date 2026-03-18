# cython: language_level=3

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()


ctypedef cnp.float64_t DTYPE_t


cdef inline double jacobi_scalar(int n, double a, double b, double x) nogil:
    cdef double p_nm2, p_nm1, p_n
    cdef double factor_1, factor_2
    cdef int k
    if n == 0:
        return 1.0
    if n == 1:
        return 0.5 * (a - b + (2.0 + a + b) * x)
    p_nm2 = 1.0
    p_nm1 = 0.5 * (a - b + (2.0 + a + b) * x)
    for k in range(2, n + 1):
        factor_1 = (
            (2.0 * k + a + b - 1.0)
            * ((a + b) * (a - b) + x * (2.0 * k + a + b - 2.0) * (2.0 * k + a + b))
            / (2.0 * (k * (k + a + b) * (2.0 * k + a + b - 2.0)))
        )
        factor_2 = (
            (k + a - 1.0) * (k + b - 1.0) * (2.0 * k + a + b)
            / (k * (k + a + b) * (2.0 * k + a + b - 2.0))
        )
        p_n = factor_1 * p_nm1 - factor_2 * p_nm2
        p_nm2 = p_nm1
        p_nm1 = p_n
    return p_nm1


def orthopoly2d_scalar(double xi, double eta, int n):
    cdef double r, s
    if eta != 1.0:
        r = 2.0 * (1.0 + xi) / (1.0 - eta) - 1.0
        s = eta
    else:
        r = -1.0
        s = 1.0
    return orthopoly2d_rst_scalar(r, s, n)


def orthopoly2d_deriv_xieta_scalar(double xi, double eta, int n):
    cdef double r, s
    if eta != 1.0:
        r = 2.0 * (1.0 + xi) / (1.0 - eta) - 1.0
        s = eta
    else:
        r = -1.0
        s = 1.0
    return orthopoly2d_deriv_rst_scalar(r, s, n)


def orthopoly2d_rst_scalar(double r, double s, int n):
    cdef int N = (n + 1) * (n + 2) // 2
    cdef cnp.ndarray[DTYPE_t, ndim=1] p = np.zeros(N, dtype=np.float64)
    cdef int ncount = 0
    cdef int ndeg, i, j
    cdef double p_i, q_i, p_j, factor
    for ndeg in range(n + 1):
        q_i = 1.0
        for i in range(ndeg + 1):
            if i == 0:
                p_i = 1.0
                q_i = 1.0
            else:
                p_i = jacobi_scalar(i, 0.0, 0.0, r)
                q_i = q_i * (1.0 - s) / 2.0
            j = ndeg - i
            if j == 0:
                p_j = 1.0
            else:
                p_j = jacobi_scalar(j, 2.0 * i + 1.0, 0.0, s)
            factor = sqrt((2.0 * i + 1.0) * (i + j + 1.0) / 2.0)
            p[ncount] = p_i * q_i * p_j * factor
            ncount += 1
    return p


def orthopoly2d_deriv_rst_scalar(double r, double s, int n):
    cdef int N = (n + 1) * (n + 2) // 2
    cdef cnp.ndarray[DTYPE_t, ndim=1] p = np.zeros(N, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] dp_dxi = np.zeros(N, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] dp_deta = np.zeros(N, dtype=np.float64)
    cdef int ncount = 0
    cdef int ndeg, i, j
    cdef double xi = (1.0 + r) * (1.0 - s) / 2.0 - 1.0
    cdef double eta = s
    cdef double dr_dxi = 2.0 / (1.0 - eta)
    cdef double dr_deta = 2.0 * (1.0 + xi) / ((1.0 - eta) * (1.0 - eta))
    cdef double p_i, q_i, dp_i, dq_i, p_j, dp_j, factor, dp_dr, dp_ds

    for ndeg in range(n + 1):
        q_i = 1.0
        for i in range(ndeg + 1):
            if i == 0:
                p_i = 1.0
                q_i = 1.0
                dp_i = 0.0
                dq_i = 0.0
            else:
                p_i = jacobi_scalar(i, 0.0, 0.0, r)
                dp_i = jacobi_scalar(i - 1, 1.0, 1.0, r) * (i + 1.0) / 2.0
                q_i = q_i * (1.0 - s) / 2.0
                dq_i = q_i * (-i) / (1.0 - s)
            j = ndeg - i
            if j == 0:
                p_j = 1.0
                dp_j = 0.0
            else:
                p_j = jacobi_scalar(j, 2.0 * i + 1.0, 0.0, s)
                dp_j = jacobi_scalar(j - 1, 2.0 * i + 2.0, 1.0, s) * (j + 2.0 * i + 2.0) / 2.0
            factor = sqrt((2.0 * i + 1.0) * (i + j + 1.0) / 2.0)
            p[ncount] = p_i * q_i * p_j * factor
            dp_dr = dp_i * q_i * p_j * factor
            dp_ds = p_i * (dq_i * p_j + q_i * dp_j) * factor
            dp_dxi[ncount] = dp_dr * dr_dxi
            dp_deta[ncount] = dp_dr * dr_deta + dp_ds
            ncount += 1
    return p, dp_dxi, dp_deta
