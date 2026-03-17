# Interpolator Benchmark Baseline

This note keeps the current benchmark baseline for
[benchmark_interpolators.py](/home/ikudashev/Documents/Github/HDG_postprocess/scripts/benchmark_interpolators.py)
and serves as the comparison point for the next optimization pass in
[interpolators.py](/home/ikudashev/Documents/Github/HDG_postprocess/hdg_postprocess/routines/interpolators.py).

## Command

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tofu-cherab
python scripts/benchmark_interpolators.py --scenario legacy_first --repeat 4 --unique-points 2000 --repeated-points 64
```

## Workload definition

- `*_cold`: first pass over a cleared cache
- `*_warm`: immediate second pass over the same cached points
- `*_mixed`: the multi-pass average over `repeat` traversals after a cache clear

The benchmark uses the real solution-loading and `define_interpolators()` path, so locator work is included.

## Current baseline

Scenario: `legacy_first`

### Cold passes

```text
value_unique_cold_seconds=0.833692
value_unique_cold_queries_per_second=2398.97

value_repeated_cold_seconds=0.028252
value_repeated_cold_queries_per_second=2265.29

gradient_unique_cold_seconds=0.837308
gradient_unique_cold_queries_per_second=2388.61

gradient_repeated_cold_seconds=0.027898
gradient_repeated_cold_queries_per_second=2294.06

mixed_unique_cold_seconds=0.850703
mixed_unique_cold_queries_per_second=2351.00

mixed_repeated_cold_seconds=0.026952
mixed_repeated_cold_queries_per_second=2374.58
```

### Warm passes

```text
value_unique_warm_seconds=0.004909
value_unique_warm_queries_per_second=407411.64

value_repeated_warm_seconds=0.000186
value_repeated_warm_queries_per_second=343783.03

gradient_unique_warm_seconds=0.008468
gradient_unique_warm_queries_per_second=236190.03

gradient_repeated_warm_seconds=0.000326
gradient_repeated_warm_queries_per_second=196078.41

mixed_unique_warm_seconds=0.013230
mixed_unique_warm_queries_per_second=151166.67

mixed_repeated_warm_seconds=0.000459
mixed_repeated_warm_queries_per_second=139303.64
```

### Mixed multi-pass results

```text
value_unique_mixed_seconds=0.827175
value_unique_mixed_queries_per_second=9671.47

value_repeated_mixed_seconds=0.027209
value_repeated_mixed_queries_per_second=9408.55

gradient_unique_mixed_seconds=0.860517
gradient_unique_mixed_queries_per_second=9296.74

gradient_repeated_mixed_seconds=0.027210
gradient_repeated_mixed_queries_per_second=9408.39

mixed_unique_mixed_seconds=0.886476
mixed_unique_mixed_queries_per_second=9024.50

mixed_repeated_mixed_seconds=0.029839
mixed_repeated_mixed_queries_per_second=8579.24
```

## Main interpretation

- Warm cached calls are dramatically faster than cold calls.
- The old mixed throughput numbers make unique and repeated workloads look more similar than they really are, because they average cold and warm phases together.
- For value interpolation, warm cached throughput is roughly two orders of magnitude higher than cold throughput.
- Gradient and mixed calls also benefit strongly from caching, but they still do more work on the warm path than value-only interpolation.
- After the structural cleanup, the safe optimization passes recovered the earlier mixed-path regression for the main unique-point workloads and kept warm cached performance strong.

## Refactor note

- `dcf279c` (`Refactor interpolator cache flow`) made the class cleaner but slightly slower.
- `58a770b` (`Optimize interpolator fast path`) recovered most of the hot-path regression and improved warm cached calls noticeably.
- The current benchmark numbers also include the follow-up allocation-reduction pass in `xieta_element()`, `xieta_element_precise()`, and `_compute_shape_data()`.

## Likely remaining bottlenecks

The cold-path cost is still dominated by the miss-side mathematical work, especially:

- `xieta_element_precise()`
- `orthopoly2D_deriv_xieta()` / `orthopoly2D_deriv_rst()`
- `jacobi()`

Those are good candidates for compiled acceleration later if more speed is needed. The failed pure-Python iterative `jacobi()` experiment was a useful sign here: this part of the stack is performance-sensitive enough that further gains may be better pursued with Cython or another compiled path rather than more aggressive Python-level rewrites.
