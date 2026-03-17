# Locator Decision Note

During Step 4 of the refactor, we explored replacing `raysect.core.math.function.float.Discrete2DMesh`
with a native locator implemented inside this repository.

## Outcome

The native prototypes were not accepted for production use.

- Exactness on clear interior and exterior points was achievable.
- However, benchmark runs on demo meshes were much slower than `raysect`.
- Representative centroid-query timings were about 53x slower than `raysect`
  for both the WEST mesh and the embedded `k` mesh.

## Decision

Keep `raysect` as the production locator baseline for now.

The benchmark script is kept in `scripts/benchmark_locator.py` as a reference
for future locator work. A native locator can be revisited after the broader
refactor, but it should be compiled or otherwise optimized enough to be
competitive with `raysect` before replacing the current implementation.

## Current detailed finding

After the interpolator speedups, the dominant end-to-end cost is no longer the interpolation kernel itself. It is now the locator setup inside `solution.sample.define_interpolators()`.

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

So the current bottleneck is specifically locator construction, not locator query speed.

## How the current locator works

The locator is built in
[geometry.py](/home/ikudashev/Documents/Github/HDG_postprocess/hdg_postprocess/core/mesh/geometry.py)
by calling `Discrete2DMesh(...)` on:

- `mesh.global_state.vertices`
- `mesh.derived_geometry.connectivity_big`
- one repeated element id per split triangle

For `legacy_first`:

- global high-order elements: `81,446`
- split triangles in `connectivity_big`: `1,303,136`
- triangles per element: `16`

So the Raysect locator is being built over roughly `1.3 million` triangles. That explains why setup is expensive even though per-query interpolation is now much faster.

Inspection of the Raysect Cython sources shows that `Discrete2DMesh` is built on a real adaptive KD-tree, not on a simple grid:

- `Discrete2DMesh` builds `MeshKDTree2D`
- `MeshKDTree2D` extends `KDTree2DCore`
- one `Item2D` is created per triangle
- each triangle gets a padded bounding box
- `KDTree2DCore` uses an SAH-style split search over candidate edges
- query traverses the tree and then does compiled barycentric triangle containment
- Raysect also keeps a tiny exact-point cache for repeated queries

So Raysect’s main advantage is very fast query traversal. Its weakness in this use case is the extremely expensive build over the full split-triangle set.

## Practical acceleration options

### 1. Keep Raysect, but cache the built locator

This is the lowest-risk improvement.

If the same solution/mesh is reused repeatedly in one process, make sure the locator is built once and then reused across:

- interpolator creation
- profile sampling
- diagnostics
- repeated analysis passes

The current API already stores the locator on `mesh.derived_geometry.element_locator`, so the main question is whether higher-level workflows rebuild it unnecessarily across sessions or scripts.

### 2. Build a custom compiled locator over high-order-element bounding boxes

This is the most promising next real optimization.

A good design would be:

- precompute one axis-aligned bounding box per high-order element
- build a spatial index over those boxes
- for a query point:
  - get candidate elements from the box index
  - run the precise local-coordinate / inside-element test only on candidates

Why this is attractive:

- build size becomes `~81k` elements instead of `~1.3M` split triangles
- the interpolator already has the precise curved-element local-coordinate logic
- the first stage can be simpler and much cheaper to build than the current Raysect triangle mesh

This should almost certainly be compiled if it is meant to compete with Raysect.

### 3. Build a native locator over coarse triangles, not over all split triangles

If a full element-box index is not the first step, another option is:

- use a much smaller triangle set for coarse candidate lookup
- then refine to the high-order element with `xieta_element_precise()`

This is conceptually similar to the box-index idea, but still triangle-based.

### 4. Add batch query support later

Batch queries will help total query throughput, but they do not solve the current dominant cost, which is build time.

So batch evaluation is still worthwhile, but it should come after locator build-time work, not before.

### 5. Persist locator artifacts

Longer-term, if locator build is still a major startup cost, one could consider serializing a native locator index to disk. That is a larger design step and probably not the first thing to do.

## Native prototype progression

### First compiled prototype: uniform grid over split triangles

The first native prototype built a uniform-grid index directly over the split triangles.

Benchmark on `legacy_mesh_west`:

- Raysect:
  - build: `52.651 s`
  - query: `0.007 s`
  - total: `52.658 s`
- Native grid prototype:
  - build: `0.193 s`
  - query: `0.116 s`
  - total: `0.308 s`

This proved that native setup time could be reduced dramatically, but query speed was too weak.

### Current compiled prototype: tree over grouped element boxes

The current `hdg_postprocess.locator.Exact2DMeshFunction` now:

- groups split triangles by their repeated high-order element id
- builds one bounding box per high-order element
- builds a compiled binary tree over those grouped element boxes
- refines inside each leaf by checking only the triangles belonging to candidate elements
- keeps a tiny exact-point cache for repeated queries

Benchmark on `legacy_mesh_west` with `scripts/benchmark_locator.py --scenario legacy_mesh_west --repeat 3`:

- Raysect:
  - build: `53.100 s`
  - query: `0.002 s`
  - total: `53.102 s`
- Native tree prototype:
  - build: `0.360 s`
  - query: `0.010 s`
  - total: `0.370 s`

Interpretation:

- build time is still dramatically better than Raysect
- query time is still slower than Raysect, but much better than the earlier grid prototype
- total end-to-end time for this workload is still overwhelmingly better for the native prototype because setup dominates so strongly in current workflows

Correctness check on the centroid-query benchmark set:

- benchmark points checked: `601`
- mismatches against Raysect: `0`

So the tree-based native prototype is a much stronger direction than the original grid version.

## Recommended next locator step

The best next step is to keep improving the tree-based native locator whose build size already scales with the number of high-order elements rather than the number of split triangles.

Suggested order:

1. tighten query traversal in the native tree
2. add better leaf sizing and split heuristics than the current median split
3. benchmark build time separately from query time
4. compare again against the Raysect KD-tree split
5. only replace Raysect in production once the native query path is strong enough for the actual workloads
