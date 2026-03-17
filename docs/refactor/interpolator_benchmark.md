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
value_unique_cold_seconds=0.828619
value_unique_cold_queries_per_second=2413.65

value_repeated_cold_seconds=0.027142
value_repeated_cold_queries_per_second=2357.97

gradient_unique_cold_seconds=0.854686
gradient_unique_cold_queries_per_second=2340.04

gradient_repeated_cold_seconds=0.027534
gradient_repeated_cold_queries_per_second=2324.38

mixed_unique_cold_seconds=0.821813
mixed_unique_cold_queries_per_second=2433.64

mixed_repeated_cold_seconds=0.026643
mixed_repeated_cold_queries_per_second=2402.15
```

### Warm passes

```text
value_unique_warm_seconds=0.004311
value_unique_warm_queries_per_second=463877.61

value_repeated_warm_seconds=0.000138
value_repeated_warm_queries_per_second=464832.73

gradient_unique_warm_seconds=0.008642
gradient_unique_warm_queries_per_second=231427.13

gradient_repeated_warm_seconds=0.000378
gradient_repeated_warm_queries_per_second=169514.90

mixed_unique_warm_seconds=0.011401
mixed_unique_warm_queries_per_second=175423.24

mixed_repeated_warm_seconds=0.000525
mixed_repeated_warm_queries_per_second=121911.02
```

### Mixed multi-pass results

```text
value_unique_mixed_seconds=0.848758
value_unique_mixed_queries_per_second=9425.54

value_repeated_mixed_seconds=0.028823
value_repeated_mixed_queries_per_second=8881.94

gradient_unique_mixed_seconds=0.882651
gradient_unique_mixed_queries_per_second=9063.60

gradient_repeated_mixed_seconds=0.029131
gradient_repeated_mixed_queries_per_second=8787.89

mixed_unique_mixed_seconds=0.886075
mixed_unique_mixed_queries_per_second=9028.58

mixed_repeated_mixed_seconds=0.032198
mixed_repeated_mixed_queries_per_second=7950.90
```

## Main interpretation

- Warm cached calls are dramatically faster than cold calls.
- The old mixed throughput numbers make unique and repeated workloads look more similar than they really are, because they average cold and warm phases together.
- For value interpolation, warm cached throughput is roughly two orders of magnitude higher than cold throughput.
- Gradient and mixed calls also benefit strongly from caching, but they still do more work on the warm path than value-only interpolation.
- After the structural cleanup, the first two optimization passes recovered the earlier mixed-path regression for the main unique-point workloads and improved the warm cached value and mixed paths further.

## Refactor note

- `dcf279c` (`Refactor interpolator cache flow`) made the class cleaner but slightly slower.
- `58a770b` (`Optimize interpolator fast path`) recovered most of the hot-path regression and improved warm cached calls noticeably.
- The current benchmark numbers also include the follow-up allocation-reduction pass in `xieta_element()`, `xieta_element_precise()`, and `_compute_shape_data()`.
