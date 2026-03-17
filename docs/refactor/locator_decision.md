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
[geometry.py](../../hdg_postprocess/core/mesh/geometry.py)
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
[locator.pyx](../../hdg_postprocess/core/mesh/locator.pyx)
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

Benchmark on `legacy_mesh_west` with `scripts/benchmark_locator.py --scenario legacy_mesh_west --leaf-sweep 8,32 --diagnostics`.

The benchmark now uses a higher default query repeat so the query numbers are less noisy than the earlier very short runs.

- Raysect:
  - build: `53.513 s`
  - query: `0.0241 s`
  - total: `53.537 s`

Representative native results:

- leaf `8`:
  - build: `0.415 s`
  - query: `0.0921 s`
  - total: `0.507 s`
- leaf `32`:
  - build: `0.199 s`
  - query: `0.0898 s`
  - total: `0.289 s`

Best observed settings in this sweep:

- best total time: leaf `32`
- best query time: leaf `32`

Correctness check on the centroid-query benchmark set:

- points checked: `601`
- mismatches against Raysect: `0`

Diagnostics from the same run explain why leaf-size tuning alone changes little:

- leaf `8`:
  - average node visits per query: `24.29`
  - average element tests per query: `5.09`
  - average triangle tests per query: `16.55`
- leaf `32`:
  - average node visits per query: `20.12`
  - average element tests per query: `14.93`
  - average triangle tests per query: `16.71`

So leaf `8` trades more traversal for fewer candidate elements, while leaf `32` trades fewer nodes for more candidate elements. In both cases the query ends up doing almost the same number of triangle tests, which is why the timing barely moves.

## Interpretation

At this point, the native tree prototype is already a credible direction:

- build time is vastly better than Raysect
- query time is still about `4x` slower than Raysect on this benchmark
- total end-to-end time is overwhelmingly better because locator build dominates current workflows

The important remaining question is not whether the native tree idea works. It does. The next question is how much more query performance can still be recovered with a better tree.

## Most Promising Next Improvements

### 1. Improve traversal and leaf refinement

The new diagnostics show that the triangle-refinement workload per query barely changes across reasonable leaf sizes. That means the next likely win is not more leaf-size tuning by itself, but reducing candidate and triangle work inside the winning leaves:

- traverse the more likely child first
- tighten candidate filtering inside leaves
- reduce repeated box checks before triangle refinement

### 2. Tune leaf size against real workloads

The diagnostics sweep shows that:

- smaller leaves do not help query much on this benchmark
- larger leaves reduce build time noticeably

So leaf size should be treated as a tuning parameter rather than fixed by intuition.

### 3. Revisit smarter split heuristics carefully

A sampled SAH-lite split was prototyped, but it increased build cost without improving query time enough to keep. Smarter splits are still a valid direction, but they need to be justified against the new diagnostics rather than assumed to help automatically.

### 4. Only then revisit batch or persistence

Batch queries and serialized locator artifacts may still be worthwhile later, but they are secondary compared with tree quality right now.

## Production Decision

The native compiled locator is now the production default in the library.

Why that decision is reasonable:

- startup time improves dramatically because locator build no longer scales with the full split-triangle KD-tree
- the broader random-point comparison against Raysect stayed clean:
  - `20,000` random points on `legacy_mesh_west`
  - `0` mismatches
  - `0` inside/outside mismatches
- the remaining query slowdown versus Raysect is acceptable for the current workflows because setup time dominates overall runtime

Raysect is still useful as an optional comparison baseline in `scripts/benchmark_locator.py`, but it is no longer required for production interpolation inside the package.

## Current Recommendation

For the locator itself, this is a good stopping point unless a larger redesign becomes worthwhile later.

If locator work is revisited in the future, the next ideas worth considering are:

1. batch-oriented query APIs rather than more scalar-query micro-tuning
2. a more substantial tree redesign, not just local heuristic tweaks
3. serialized locator artifacts if startup time becomes critical across repeated short-lived processes
