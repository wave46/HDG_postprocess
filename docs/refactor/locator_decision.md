# Locator Decision Note

During the interpolator optimization pass, the dominant setup cost moved away from the interpolation kernels and into locator construction. This note records what the current locator does, how Raysect works, and what the native prototype shows so far.

## Current Bottleneck

Measured on `legacy_first`:

- `load_solution`: about `1.61 s`
- `define_interpolators`: about `54.22 s`
- total setup: about `55.82 s`

Breaking `define_interpolators()` down further:

- `ensure_simple_solution`: about `0.54 s`
- `ensure_connectivity_big`: about `0.05 s`
- `element_locator`: about `54.30 s`
- `define_qcyl`: about `0.11 s`
- remaining interpolator instance setup after prerequisites: about `0.20 s`

So the main startup bottleneck is locator construction, not interpolation calls anymore.

## Raysect Baseline

The current production locator is built in
[geometry.py](/home/ikudashev/Documents/Github/HDG_postprocess/hdg_postprocess/core/mesh/geometry.py)
with `raysect.core.math.function.float.Discrete2DMesh(...)`.

For `legacy_first`:

- global high-order elements: `81,446`
- split triangles in `connectivity_big`: `1,303,136`
- triangles per element: `16`

Inspection of the installed Raysect Cython sources shows that `Discrete2DMesh` is backed by a real KD-tree:

- `Discrete2DMesh` builds `MeshKDTree2D`
- `MeshKDTree2D` extends `KDTree2DCore`
- one item is inserted per split triangle
- each triangle gets a padded bounding box
- the builder uses an SAH-like split search over candidate edges
- query traversal ends with compiled barycentric triangle containment
- Raysect also keeps a tiny exact-point cache

This explains the current tradeoff:

- query is very fast
- build is very expensive because the tree is built over the full split-triangle set

## Native Locator Structure

The native prototype in
[locator.pyx](/home/ikudashev/Documents/Github/HDG_postprocess/hdg_postprocess/locator.pyx)
now uses a tree over grouped high-order elements rather than over all split triangles.

The structure is:

1. Sort split triangles by repeated high-order element id.
2. Group consecutive triangles that belong to the same high-order element.
3. Build one axis-aligned bounding box per high-order element.
4. Build a binary tree over those element boxes.
5. On query:
   - traverse the tree by node bounding boxes
   - test only candidate element boxes in leaf nodes
   - refine with point-in-triangle checks over that element's split triangles
6. Keep a tiny exact-point cache for repeated calls.

Internally the tree is stored as a structure-of-arrays:

- `node_bounds`
- `node_left`
- `node_right`
- `node_leaf_start`
- `node_leaf_count`
- `leaf_indices`

This layout is simple for Cython, keeps traversal compact, and is easier to optimize than a Python object tree.

## Prototype Evolution

### First native prototype: uniform grid over split triangles

Benchmark on `legacy_mesh_west`:

- Raysect:
  - build: `52.651 s`
  - query: `0.007 s`
  - total: `52.658 s`
- Native grid prototype:
  - build: `0.193 s`
  - query: `0.116 s`
  - total: `0.308 s`

This proved that setup time could be reduced dramatically, but the query path was too weak.

### Current native prototype: tree over grouped element boxes

Benchmark on `legacy_mesh_west` with `scripts/benchmark_locator.py --scenario legacy_mesh_west --leaf-sweep 4,8,16,32`.

The benchmark now uses a higher default query repeat so the query numbers are less noisy than the earlier very short runs.

- Raysect:
  - build: `53.558 s`
  - query: `0.0177 s`
  - total: `53.576 s`

Leaf-size sweep for the native tree:

- leaf `4`:
  - build: `0.629 s`
  - query: `0.0911 s`
  - total: `0.720 s`
- leaf `8`:
  - build: `0.387 s`
  - query: `0.0896 s`
  - total: `0.476 s`
- leaf `16`:
  - build: `0.256 s`
  - query: `0.0903 s`
  - total: `0.346 s`
- leaf `32`:
  - build: `0.197 s`
  - query: `0.0895 s`
  - total: `0.287 s`

Best observed settings in this sweep:

- best total time: leaf `32`
- best query time: leaf `32`

Correctness check on the centroid-query benchmark set:

- points checked: `601`
- mismatches against Raysect: `0`

## Interpretation

At this point, the native tree prototype is already a credible direction:

- build time is vastly better than Raysect
- query time is still about `5x` slower than Raysect on this benchmark
- total end-to-end time is overwhelmingly better because locator build dominates current workflows

The important remaining question is not whether the native tree idea works. It does. The next question is how much more query performance can still be recovered with a better tree.

## Most Promising Next Improvements

### 1. Improve split quality

The current tree still uses a simple median split on the longest axis. The next meaningful improvement is a better split heuristic, for example an SAH-lite score:

- evaluate several candidate split positions
- score each split with something like
  - `left_count * left_area + right_count * right_area`
- choose the lowest-cost split

This is the closest low-risk step toward the quality of the Raysect KD-tree.

### 2. Tune leaf size against real workloads

The first sweep shows that:

- smaller leaves do not help query much on this benchmark
- larger leaves reduce build time noticeably

So leaf size should be treated as a tuning parameter rather than fixed by intuition.

### 3. Tighten leaf refinement

If needed later:

- store tighter leaf-local triangle ranges
- improve traversal order
- reduce unnecessary box checks inside leaves

### 4. Only then revisit batch or persistence

Batch queries and serialized locator artifacts may still be worthwhile later, but they are secondary compared with tree quality right now.

## Current Decision

Keep Raysect as the production default for the moment, but the native tree prototype is now strong enough to justify continued development.

The best next locator step is:

1. keep the current grouped-element tree
2. replace the median split with a better split heuristic
3. re-benchmark build and query separately
4. compare again against Raysect on both setup-heavy and query-heavy workloads
