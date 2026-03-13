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
