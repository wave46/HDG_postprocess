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
value_unique_cold_seconds=0.845999
value_unique_cold_queries_per_second=2364.07

value_repeated_cold_seconds=0.028365
value_repeated_cold_queries_per_second=2256.27

gradient_unique_cold_seconds=0.858129
gradient_unique_cold_queries_per_second=2330.65

gradient_repeated_cold_seconds=0.028508
gradient_repeated_cold_queries_per_second=2244.95

mixed_unique_cold_seconds=0.847226
mixed_unique_cold_queries_per_second=2360.65

mixed_repeated_cold_seconds=0.027795
mixed_repeated_cold_queries_per_second=2302.58
```

### Warm passes

```text
value_unique_warm_seconds=0.004585
value_unique_warm_queries_per_second=436188.47

value_repeated_warm_seconds=0.000147
value_repeated_warm_queries_per_second=436755.75

gradient_unique_warm_seconds=0.007644
gradient_unique_warm_queries_per_second=261651.67

gradient_repeated_warm_seconds=0.000403
gradient_repeated_warm_queries_per_second=158952.91

mixed_unique_warm_seconds=0.012282
mixed_unique_warm_queries_per_second=162844.13

mixed_repeated_warm_seconds=0.000418
mixed_repeated_warm_queries_per_second=153173.08
```

### Mixed multi-pass results

```text
value_unique_mixed_seconds=0.851681
value_unique_mixed_queries_per_second=9393.19

value_repeated_mixed_seconds=0.028700
value_repeated_mixed_queries_per_second=8919.88

gradient_unique_mixed_seconds=0.841244
gradient_unique_mixed_queries_per_second=9509.73

gradient_repeated_mixed_seconds=0.029293
gradient_repeated_mixed_queries_per_second=8739.28

mixed_unique_mixed_seconds=0.898281
mixed_unique_mixed_queries_per_second=8905.90

mixed_repeated_mixed_seconds=0.030149
mixed_repeated_mixed_queries_per_second=8491.29
```

## Main interpretation

- Warm cached calls are dramatically faster than cold calls.
- The old mixed throughput numbers make unique and repeated workloads look more similar than they really are, because they average cold and warm phases together.
- For value interpolation, warm cached throughput is roughly two orders of magnitude higher than cold throughput.
- Gradient and mixed calls also benefit strongly from caching, but they still do more work on the warm path than value-only interpolation.
- After the first cleanup and this first hot-path optimization pass, the mixed-path regression from the structural refactor is mostly recovered while warm-cache performance is clearly better.

## Refactor note

- `dcf279c` (`Refactor interpolator cache flow`) made the class cleaner but slightly slower.
- The current benchmark numbers include the follow-up hot-path optimization pass, which recovers most of that slowdown and improves warm cached calls noticeably.
