# IMAS Export Performance Notes

## Baseline

- Branch: `feature/imas-export`
- Environment: `hdg-postprocess-py312`
- Representative case:
  - `demos/data/solutions/power_balance_boundary_checks/solution_west_heating_cooling_factor`
  - repeated as same-mesh snapshots to isolate exporter cost

## Timings

### 8 snapshots, `dr = dz = 0.01`

- grid: `138 x 175`
- points: `24,150`
- `build_timed_snapshots`: `7.964 s`
- first equilibrium sampling: `1.942 s`
- first plasma sampling: `0.805 s`
- `build_summary_ids`: `1.953 s`
- `build_plasma_ids`: `35.620 s`
- `build_equilibrium_ids`: `18.407 s`
- `put_summary`: `0.096 s`
- `put_plasma`: `23.806 s`
- `put_equilibrium`: `0.322 s`
- output size: `16.16 MB`

### 4 snapshots, `dr = dz = 0.005`

- grid: `274 x 349`
- points: `95,626`
- `build_timed_snapshots`: `6.485 s`
- first equilibrium sampling: `6.695 s`
- first plasma sampling: `1.957 s`
- `build_summary_ids`: `1.615 s`
- `build_plasma_ids`: `47.068 s`
- `build_equilibrium_ids`: `18.045 s`
- `put_summary`: `0.018 s`
- `put_plasma`: `46.820 s`
- `put_equilibrium`: `0.872 s`
- output size: `45.51 MB`

## Bottleneck

Plasma export dominates runtime.

From `cProfile` on `4` snapshots at `dr = dz = 0.005`:

- plasma build total: `84.163 s`
- equilibrium build total: `13.791 s`

Main plasma hotspot:

- `populate_rectangular_grid_ggd_entry(...)`: about `56.9 s`

Secondary plasma hotspot:

- `sample_plasma_fields(...)`: about `15.1 s`

Main equilibrium hotspot:

- `sample_equilibrium_fields(...)`: about `6.1 s`

## Current Conclusion

- The main cost is rectangular plasma GGD construction and writing.
- Plasma sampling is the next significant cost.
- Equilibrium is materially cheaper than plasma.
- For large discharges, the best acceleration target is reducing explicit plasma GGD volume before further interpolator micro-optimization.

## Tried: cached rectangular topology data

Attempt:

- cache `rectangular_topology(grid)` by grid signature
- reuse the same plain node/edge/cell lists across equilibrium and plasma
- reuse them again for unchanged grids over time

Result on the fine WEST-like grid (`dr = dz = 0.005`):

- topology build itself:
  - cold: `0.503 s`
  - warm: `0.000 s`
- but `build_plasma_ids` for `4` snapshots still stayed around:
  - `48.625 s`

Conclusion:

- caching the plain topology data works
- but it is not the real bottleneck
- the expensive part is populating IMAS objects with that topology, not generating the node/edge/cell lists

Implication:

- a stronger optimization is needed:
  - either reusing unchanged `grid_ggd` semantically by IMAS reference/path
  - or otherwise avoiding repeated IMAS topology population itself

## Tried: cached topology payload arrays

Attempt:

- precompute not only node/edge/cell lists
- but also the NumPy payload objects later assigned into IMAS node, edge, cell, and subset entries

Micro-benchmark on the fine WEST-like grid (`dr = dz = 0.005`):

- current grid population: about `7.0 s`
- payload-based population: about `6.3 s`

End-to-end check on `build_plasma_ids` for `4` snapshots:

- before: about `48.6 s`
- payload-based version: about `47.8 s`

Conclusion:

- the idea helps slightly
- but the gain is too small to justify extra complexity
- the main bottleneck is still repeated IMAS object population itself

Decision:

- do not keep the payload-cache approach
- move on to unchanged-grid reuse by IMAS `path` / reference

## Tried: unchanged-grid reuse by IMAS `path`

Attempt:

- in `plasma_profiles.grid_ggd`, store the first explicit grid topology for each unique rectangular grid
- for later unchanged snapshots, store only a `path` reference to that first explicit `grid_ggd`
- keep `equilibrium.grids_ggd` pointing directly to the corresponding explicit plasma grid entry

Result on `4` repeated snapshots with `dr = dz = 0.005`:

- previous `build_plasma_ids`: `47.068 s`
- path-reuse `build_plasma_ids`: `27.610 s`
- `build_equilibrium_ids`: `11.515 s`

Roundtrip check:

- readback keeps the expected plasma paths:
  - `["", "#plasma_profiles:0/grid_ggd(1)", "#plasma_profiles:0/grid_ggd(1)", "#plasma_profiles:0/grid_ggd(1)"]`
- equilibrium keeps direct references to the explicit plasma grid:
  - `["#plasma_profiles:0/grid_ggd(1)", ...]`

Mixed-grid check:

- when the grid changes every 3 timesteps, the exporter produces:
  - plasma: `[None, ref(1), ref(1), None, ref(4), ref(4), None, ref(7), ref(7)]`
  - equilibrium: `[ref(1), ref(1), ref(1), ref(4), ref(4), ref(4), ref(7), ref(7), ref(7)]`

Conclusion:

- unchanged-grid reuse by IMAS `path` is the first optimization that materially reduces plasma build cost
- this should be the default discharge-export behavior when consecutive rectangular grids are identical
