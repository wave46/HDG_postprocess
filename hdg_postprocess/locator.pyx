# cython: language_level=3

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt


ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t ITYPE_t


cdef class Exact2DMeshFunction:
    cdef object _vertices
    cdef object _connectivity
    cdef object _values
    cdef list _cells
    cdef double _min_x
    cdef double _max_x
    cdef double _min_y
    cdef double _max_y
    cdef double _dx
    cdef double _dy
    cdef double _default_value
    cdef bint _limit
    cdef int _bins_x
    cdef int _bins_y

    def __init__(
        self,
        vertices,
        connectivity,
        values,
        limit=False,
        default_value=-1,
        bins_x=0,
        bins_y=0,
        target_triangles_per_cell=16,
    ):
        cdef Py_ssize_t n_triangles
        cdef double span_x, span_y, aspect, approx_bins

        self._vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self._connectivity = np.ascontiguousarray(connectivity, dtype=np.int64)
        self._values = np.ascontiguousarray(values, dtype=np.float64)
        self._limit = limit
        self._default_value = float(default_value)

        self._min_x = float(np.min(self._vertices[:, 0]))
        self._max_x = float(np.max(self._vertices[:, 0]))
        self._min_y = float(np.min(self._vertices[:, 1]))
        self._max_y = float(np.max(self._vertices[:, 1]))
        span_x = max(self._max_x - self._min_x, 1e-12)
        span_y = max(self._max_y - self._min_y, 1e-12)
        n_triangles = self._connectivity.shape[0]

        if bins_x <= 0 or bins_y <= 0:
            aspect = sqrt(span_x / span_y)
            approx_bins = sqrt(n_triangles / max(target_triangles_per_cell, 1))
            bins_x = max(1, int(approx_bins * aspect))
            bins_y = max(1, int(approx_bins / aspect))

        self._bins_x = bins_x
        self._bins_y = bins_y
        self._dx = span_x / self._bins_x
        self._dy = span_y / self._bins_y
        self._cells = [list() for _ in range(self._bins_x * self._bins_y)]
        self._build_cells()

    cdef int _clip_ix(self, int ix):
        if ix < 0:
            return 0
        if ix >= self._bins_x:
            return self._bins_x - 1
        return ix

    cdef int _clip_iy(self, int iy):
        if iy < 0:
            return 0
        if iy >= self._bins_y:
            return self._bins_y - 1
        return iy

    cdef int _cell_index(self, double x, double y):
        cdef int ix = self._clip_ix(<int>((x - self._min_x) / self._dx))
        cdef int iy = self._clip_iy(<int>((y - self._min_y) / self._dy))
        return iy * self._bins_x + ix

    cdef void _build_cells(self):
        cdef Py_ssize_t tri_id
        cdef int ix0, ix1, iy0, iy1, ix, iy
        cdef ITYPE_t n0, n1, n2
        cdef double x0, x1, x2, y0, y1, y2
        cdef double min_x, max_x, min_y, max_y
        cdef DTYPE_t[:, ::1] vertices = self._vertices
        cdef ITYPE_t[:, ::1] connectivity = self._connectivity

        for tri_id in range(connectivity.shape[0]):
            n0 = connectivity[tri_id, 0]
            n1 = connectivity[tri_id, 1]
            n2 = connectivity[tri_id, 2]

            x0 = vertices[n0, 0]
            x1 = vertices[n1, 0]
            x2 = vertices[n2, 0]
            y0 = vertices[n0, 1]
            y1 = vertices[n1, 1]
            y2 = vertices[n2, 1]

            min_x = min(x0, x1, x2)
            max_x = max(x0, x1, x2)
            min_y = min(y0, y1, y2)
            max_y = max(y0, y1, y2)

            ix0 = self._clip_ix(<int>((min_x - self._min_x) / self._dx))
            ix1 = self._clip_ix(<int>((max_x - self._min_x) / self._dx))
            iy0 = self._clip_iy(<int>((min_y - self._min_y) / self._dy))
            iy1 = self._clip_iy(<int>((max_y - self._min_y) / self._dy))

            for iy in range(iy0, iy1 + 1):
                for ix in range(ix0, ix1 + 1):
                    self._cells[iy * self._bins_x + ix].append(tri_id)

    cdef bint _point_in_triangle(self, Py_ssize_t tri_id, double x, double y):
        cdef DTYPE_t[:, ::1] vertices = self._vertices
        cdef ITYPE_t[:, ::1] connectivity = self._connectivity
        cdef ITYPE_t n0 = connectivity[tri_id, 0]
        cdef ITYPE_t n1 = connectivity[tri_id, 1]
        cdef ITYPE_t n2 = connectivity[tri_id, 2]
        cdef double x0 = vertices[n0, 0]
        cdef double y0 = vertices[n0, 1]
        cdef double x1 = vertices[n1, 0]
        cdef double y1 = vertices[n1, 1]
        cdef double x2 = vertices[n2, 0]
        cdef double y2 = vertices[n2, 1]
        cdef double det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        cdef double a, b, c
        cdef double eps = 1e-12
        if det == 0.0:
            return False
        a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / det
        b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / det
        c = 1.0 - a - b
        return a >= -eps and b >= -eps and c >= -eps

    def __call__(self, double x, double y):
        cdef list cell
        cdef Py_ssize_t tri_id
        cdef int cell_index
        cdef DTYPE_t[::1] values = self._values

        if x < self._min_x or x > self._max_x or y < self._min_y or y > self._max_y:
            if self._limit:
                raise ValueError("Requested value outside mesh bounds.")
            return self._default_value

        cell_index = self._cell_index(x, y)
        cell = self._cells[cell_index]
        for tri_id in cell:
            if self._point_in_triangle(tri_id, x, y):
                return values[tri_id]

        if self._limit:
            raise ValueError("Requested value outside mesh bounds.")
        return self._default_value
