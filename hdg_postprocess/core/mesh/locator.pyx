# cython: language_level=3

import numpy as np
cimport numpy as cnp


ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t ITYPE_t
ctypedef cnp.int32_t INDEX_t

cdef inline bint _bbox_contains(
    DTYPE_t[:, ::1] bounds,
    INDEX_t idx,
    double x,
    double y,
):
    return (
        x >= bounds[idx, 0]
        and x <= bounds[idx, 2]
        and y >= bounds[idx, 1]
        and y <= bounds[idx, 3]
    )


cdef class Exact2DMeshFunction:
    cdef object _vertices
    cdef object _connectivity
    cdef object _element_values
    cdef object _triangle_starts
    cdef object _triangle_counts
    cdef object _element_boxes
    cdef object _element_centers
    cdef object _node_bounds
    cdef object _node_left
    cdef object _node_right
    cdef object _node_leaf_start
    cdef object _node_leaf_count
    cdef object _leaf_indices
    cdef double _default_value
    cdef bint _limit
    cdef bint _cache_available
    cdef double _cached_x
    cdef double _cached_y
    cdef double _cached_value
    cdef bint _cached_hit
    cdef int _leaf_size
    cdef bint _collect_stats
    cdef long long _query_count
    cdef long long _node_visit_count
    cdef long long _leaf_visit_count
    cdef long long _element_test_count
    cdef long long _triangle_test_count
    cdef long long _cache_hit_count

    def __init__(
        self,
        vertices,
        connectivity,
        values,
        limit=False,
        default_value=-1,
        leaf_size=8,
        collect_stats=False,
    ):
        self._limit = limit
        self._default_value = float(default_value)
        self._leaf_size = max(1, int(leaf_size))
        self._collect_stats = collect_stats
        self._prepare_mesh(vertices, connectivity, values)
        self._prepare_tree()
        self._reset_cache()
        self.reset_stats()

    cdef void _prepare_mesh(self, object vertices, object connectivity, object values):
        cdef cnp.ndarray[DTYPE_t, ndim=2] vertices_arr
        cdef cnp.ndarray[ITYPE_t, ndim=2] connectivity_arr
        cdef cnp.ndarray[ITYPE_t, ndim=1] values_arr
        cdef cnp.ndarray[ITYPE_t, ndim=1] order
        cdef cnp.ndarray[ITYPE_t, ndim=1] starts
        cdef cnp.ndarray[ITYPE_t, ndim=1] counts
        cdef cnp.ndarray[ITYPE_t, ndim=1] unique_values

        vertices_arr = np.ascontiguousarray(vertices, dtype=np.float64)
        connectivity_arr = np.ascontiguousarray(connectivity, dtype=np.int64)
        values_arr = np.ascontiguousarray(values, dtype=np.int64)
        order = np.argsort(values_arr, kind="mergesort").astype(np.int64)
        self._vertices = vertices_arr
        self._connectivity = np.ascontiguousarray(connectivity_arr[order], dtype=np.int64)
        values_arr = np.ascontiguousarray(values_arr[order], dtype=np.int64)

        unique_values, starts, counts = self._group_values(values_arr)
        self._element_values = unique_values
        self._triangle_starts = starts
        self._triangle_counts = counts
        self._element_boxes = self._build_element_boxes()
        self._element_centers = self._build_element_centers()

    cdef void _prepare_tree(self):
        cdef cnp.ndarray[INDEX_t, ndim=1] element_ids
        cdef list node_bounds = []
        cdef list node_left = []
        cdef list node_right = []
        cdef list node_leaf_start = []
        cdef list node_leaf_count = []
        cdef list leaf_indices = []

        if self._element_values is None or self._element_values.shape[0] == 0:
            self._node_bounds = np.empty((0, 4), dtype=np.float64)
            self._node_left = np.empty(0, dtype=np.int32)
            self._node_right = np.empty(0, dtype=np.int32)
            self._node_leaf_start = np.empty(0, dtype=np.int32)
            self._node_leaf_count = np.empty(0, dtype=np.int32)
            self._leaf_indices = np.empty(0, dtype=np.int32)
            return

        element_ids = np.arange(self._element_values.shape[0], dtype=np.int32)
        self._build_tree(
            element_ids,
            node_bounds,
            node_left,
            node_right,
            node_leaf_start,
            node_leaf_count,
            leaf_indices,
        )

        self._node_bounds = np.ascontiguousarray(node_bounds, dtype=np.float64)
        self._node_left = np.ascontiguousarray(node_left, dtype=np.int32)
        self._node_right = np.ascontiguousarray(node_right, dtype=np.int32)
        self._node_leaf_start = np.ascontiguousarray(node_leaf_start, dtype=np.int32)
        self._node_leaf_count = np.ascontiguousarray(node_leaf_count, dtype=np.int32)
        self._leaf_indices = np.ascontiguousarray(leaf_indices, dtype=np.int32)

    cdef void _reset_cache(self):
        self._cache_available = False
        self._cached_x = 0.0
        self._cached_y = 0.0
        self._cached_value = self._default_value
        self._cached_hit = False

    def reset_stats(self):
        self._query_count = 0
        self._node_visit_count = 0
        self._leaf_visit_count = 0
        self._element_test_count = 0
        self._triangle_test_count = 0
        self._cache_hit_count = 0

    def statistics(self):
        cdef double queries = float(self._query_count) if self._query_count > 0 else 1.0
        return {
            "queries": int(self._query_count),
            "cache_hits": int(self._cache_hit_count),
            "node_visits": int(self._node_visit_count),
            "leaf_visits": int(self._leaf_visit_count),
            "element_tests": int(self._element_test_count),
            "triangle_tests": int(self._triangle_test_count),
            "avg_node_visits_per_query": self._node_visit_count / queries,
            "avg_leaf_visits_per_query": self._leaf_visit_count / queries,
            "avg_element_tests_per_query": self._element_test_count / queries,
            "avg_triangle_tests_per_query": self._triangle_test_count / queries,
        }

    cdef object _return_default(self, double x, double y):
        self._cache_available = True
        self._cached_x = x
        self._cached_y = y
        self._cached_value = self._default_value
        self._cached_hit = False
        if self._limit:
            raise ValueError("Requested value outside mesh bounds.")
        return self._default_value

    cdef ITYPE_t _cache_hit_value(self):
        if self._cached_hit:
            return <ITYPE_t>self._cached_value
        if self._limit:
            raise ValueError("Requested value outside mesh bounds.")
        return <ITYPE_t>self._default_value

    cdef ITYPE_t _return_hit(self, double x, double y, ITYPE_t value):
        self._cache_available = True
        self._cached_x = x
        self._cached_y = y
        self._cached_value = <double>value
        self._cached_hit = True
        return value

    def _group_values(self, cnp.ndarray[ITYPE_t, ndim=1] values_arr):
        cdef Py_ssize_t n = values_arr.shape[0]
        if n == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )
        change = np.empty(n, dtype=np.bool_)
        change[0] = True
        if n > 1:
            change[1:] = values_arr[1:] != values_arr[:-1]
        starts = np.flatnonzero(change).astype(np.int64)
        counts = np.diff(np.append(starts, n)).astype(np.int64)
        unique_values = values_arr[starts].astype(np.int64)
        return unique_values, starts, counts

    cdef object _build_element_boxes(self):
        cdef DTYPE_t[:, ::1] vertices = self._vertices
        cdef ITYPE_t[:, ::1] connectivity = self._connectivity
        cdef ITYPE_t[::1] starts = self._triangle_starts
        cdef ITYPE_t[::1] counts = self._triangle_counts
        cdef cnp.ndarray[DTYPE_t, ndim=2] boxes = np.empty((starts.shape[0], 4), dtype=np.float64)
        cdef Py_ssize_t elem_id, tri_id, tri_end
        cdef ITYPE_t n0, n1, n2
        cdef double x0, x1, x2, y0, y1, y2
        cdef double min_x, max_x, min_y, max_y

        for elem_id in range(starts.shape[0]):
            tri_id = starts[elem_id]
            tri_end = tri_id + counts[elem_id]

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

            tri_id += 1
            while tri_id < tri_end:
                n0 = connectivity[tri_id, 0]
                n1 = connectivity[tri_id, 1]
                n2 = connectivity[tri_id, 2]
                x0 = vertices[n0, 0]
                x1 = vertices[n1, 0]
                x2 = vertices[n2, 0]
                y0 = vertices[n0, 1]
                y1 = vertices[n1, 1]
                y2 = vertices[n2, 1]
                min_x = min(min_x, x0, x1, x2)
                max_x = max(max_x, x0, x1, x2)
                min_y = min(min_y, y0, y1, y2)
                max_y = max(max_y, y0, y1, y2)
                tri_id += 1

            boxes[elem_id, 0] = min_x
            boxes[elem_id, 1] = min_y
            boxes[elem_id, 2] = max_x
            boxes[elem_id, 3] = max_y

        return boxes

    cdef object _build_element_centers(self):
        cdef DTYPE_t[:, ::1] boxes = self._element_boxes
        cdef cnp.ndarray[DTYPE_t, ndim=2] centers = np.empty((boxes.shape[0], 2), dtype=np.float64)
        cdef Py_ssize_t i
        for i in range(boxes.shape[0]):
            centers[i, 0] = 0.5 * (boxes[i, 0] + boxes[i, 2])
            centers[i, 1] = 0.5 * (boxes[i, 1] + boxes[i, 3])
        return centers

    cdef INDEX_t _build_tree(
        self,
        object element_ids_obj,
        list node_bounds,
        list node_left,
        list node_right,
        list node_leaf_start,
        list node_leaf_count,
        list leaf_indices,
    ):
        cdef DTYPE_t[:, ::1] boxes = self._element_boxes
        cdef DTYPE_t[:, ::1] centers = self._element_centers
        cdef cnp.ndarray[INDEX_t, ndim=1] element_ids_arr = np.ascontiguousarray(element_ids_obj, dtype=np.int32)
        cdef INDEX_t[::1] element_ids = element_ids_arr
        cdef Py_ssize_t idx, n = element_ids.shape[0]
        cdef INDEX_t node_id, left_id, right_id
        cdef double min_x, min_y, max_x, max_y
        cdef int axis
        cdef cnp.ndarray[DTYPE_t, ndim=1] split_values
        cdef cnp.ndarray[INDEX_t, ndim=1] order
        cdef cnp.ndarray[INDEX_t, ndim=1] left_ids
        cdef cnp.ndarray[INDEX_t, ndim=1] right_ids

        min_x = boxes[element_ids[0], 0]
        min_y = boxes[element_ids[0], 1]
        max_x = boxes[element_ids[0], 2]
        max_y = boxes[element_ids[0], 3]
        for idx in range(1, n):
            min_x = min(min_x, boxes[element_ids[idx], 0])
            min_y = min(min_y, boxes[element_ids[idx], 1])
            max_x = max(max_x, boxes[element_ids[idx], 2])
            max_y = max(max_y, boxes[element_ids[idx], 3])

        node_id = len(node_bounds)
        node_bounds.append((min_x, min_y, max_x, max_y))
        node_left.append(-1)
        node_right.append(-1)
        node_leaf_start.append(-1)
        node_leaf_count.append(0)

        if n <= self._leaf_size:
            node_leaf_start[node_id] = len(leaf_indices)
            node_leaf_count[node_id] = n
            leaf_indices.extend(int(v) for v in element_ids)
            return node_id

        axis = self._choose_split_axis(min_x, min_y, max_x, max_y)
        split_values = np.ascontiguousarray(np.asarray(centers)[element_ids_arr, axis], dtype=np.float64)
        order = np.argsort(split_values, kind="mergesort").astype(np.int32)
        element_ids_arr = np.ascontiguousarray(element_ids_arr[order], dtype=np.int32)
        left_ids = element_ids_arr[: n // 2]
        right_ids = element_ids_arr[n // 2 :]

        left_id = self._build_tree(left_ids, node_bounds, node_left, node_right, node_leaf_start, node_leaf_count, leaf_indices)
        right_id = self._build_tree(right_ids, node_bounds, node_left, node_right, node_leaf_start, node_leaf_count, leaf_indices)
        node_left[node_id] = left_id
        node_right[node_id] = right_id
        return node_id

    cdef int _choose_split_axis(self, double min_x, double min_y, double max_x, double max_y):
        if (max_x - min_x) >= (max_y - min_y):
            return 0
        return 1

    cdef bint _point_in_triangle(
        self,
        DTYPE_t[:, ::1] vertices,
        ITYPE_t[:, ::1] connectivity,
        ITYPE_t tri_id,
        double x,
        double y,
    ):
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

    cdef ITYPE_t _query_leaf(
        self,
        DTYPE_t[:, ::1] boxes,
        DTYPE_t[:, ::1] vertices,
        ITYPE_t[:, ::1] connectivity,
        ITYPE_t[::1] element_values,
        ITYPE_t[::1] triangle_starts,
        ITYPE_t[::1] triangle_counts,
        INDEX_t[::1] leaf_indices,
        INDEX_t leaf_start,
        INDEX_t leaf_count,
        double x,
        double y,
    ):
        cdef INDEX_t elem_id
        cdef ITYPE_t tri_id, tri_end
        cdef INDEX_t leaf_idx

        for leaf_idx in range(leaf_start, leaf_start + leaf_count):
            if self._collect_stats:
                self._element_test_count += 1
            elem_id = leaf_indices[leaf_idx]
            if not _bbox_contains(boxes, elem_id, x, y):
                continue
            tri_id = triangle_starts[elem_id]
            tri_end = tri_id + triangle_counts[elem_id]
            while tri_id < tri_end:
                if self._collect_stats:
                    self._triangle_test_count += 1
                if self._point_in_triangle(vertices, connectivity, tri_id, x, y):
                    return element_values[elem_id]
                tri_id += 1
        return <ITYPE_t>-1

    def __call__(self, double x, double y):
        cdef DTYPE_t[:, ::1] boxes = self._element_boxes
        cdef DTYPE_t[:, ::1] node_bounds = self._node_bounds
        cdef DTYPE_t[:, ::1] vertices = self._vertices
        cdef ITYPE_t[:, ::1] connectivity = self._connectivity
        cdef ITYPE_t[::1] element_values = self._element_values
        cdef ITYPE_t[::1] triangle_starts = self._triangle_starts
        cdef ITYPE_t[::1] triangle_counts = self._triangle_counts
        cdef INDEX_t[::1] node_left = self._node_left
        cdef INDEX_t[::1] node_right = self._node_right
        cdef INDEX_t[::1] node_leaf_start = self._node_leaf_start
        cdef INDEX_t[::1] node_leaf_count = self._node_leaf_count
        cdef INDEX_t[::1] leaf_indices = self._leaf_indices
        cdef INDEX_t stack[128]
        cdef int stack_size = 0
        cdef INDEX_t node_id
        cdef ITYPE_t value = -1

        if self._collect_stats:
            self._query_count += 1

        if self._cache_available and x == self._cached_x and y == self._cached_y:
            if self._collect_stats:
                self._cache_hit_count += 1
            return self._cache_hit_value()

        if node_bounds.shape[0] == 0:
            return self._return_default(x, y)

        if not _bbox_contains(node_bounds, 0, x, y):
            return self._return_default(x, y)

        stack[0] = 0
        stack_size = 1
        while stack_size > 0:
            stack_size -= 1
            node_id = stack[stack_size]
            if self._collect_stats:
                self._node_visit_count += 1
            if not _bbox_contains(node_bounds, node_id, x, y):
                continue

            if node_leaf_count[node_id] > 0:
                if self._collect_stats:
                    self._leaf_visit_count += 1
                value = self._query_leaf(
                    boxes,
                    vertices,
                    connectivity,
                    element_values,
                    triangle_starts,
                    triangle_counts,
                    leaf_indices,
                    node_leaf_start[node_id],
                    node_leaf_count[node_id],
                    x,
                    y,
                )
                if value != -1:
                    return self._return_hit(x, y, value)
                continue

            if node_left[node_id] >= 0:
                stack[stack_size] = node_left[node_id]
                stack_size += 1
            if node_right[node_id] >= 0:
                stack[stack_size] = node_right[node_id]
                stack_size += 1

        return self._return_default(x, y)
