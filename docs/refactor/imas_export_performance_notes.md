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
