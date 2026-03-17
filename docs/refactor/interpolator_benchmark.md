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
value_unique_cold_seconds=0.841032
value_unique_cold_queries_per_second=2378.03

value_repeated_cold_seconds=0.027907
value_repeated_cold_queries_per_second=2293.31

gradient_unique_cold_seconds=0.859970
gradient_unique_cold_queries_per_second=2325.66

gradient_repeated_cold_seconds=0.028112
gradient_repeated_cold_queries_per_second=2276.59

mixed_unique_cold_seconds=0.896586
mixed_unique_cold_queries_per_second=2230.68

mixed_repeated_cold_seconds=0.032009
mixed_repeated_cold_queries_per_second=1999.46
```

### Warm passes

```text
value_unique_warm_seconds=0.005853
value_unique_warm_queries_per_second=341696.17

value_repeated_warm_seconds=0.000187
value_repeated_warm_queries_per_second=342388.75

gradient_unique_warm_seconds=0.010783
gradient_unique_warm_queries_per_second=185484.64

gradient_repeated_warm_seconds=0.000379
gradient_repeated_warm_queries_per_second=168714.95

mixed_unique_warm_seconds=0.017174
mixed_unique_warm_queries_per_second=116454.57

mixed_repeated_warm_seconds=0.000614
mixed_repeated_warm_queries_per_second=104270.21
```

### Mixed multi-pass results

```text
value_unique_mixed_seconds=0.872814
value_unique_mixed_queries_per_second=9165.75

value_repeated_mixed_seconds=0.029157
value_repeated_mixed_queries_per_second=8780.04

gradient_unique_mixed_seconds=0.889470
gradient_unique_mixed_queries_per_second=8994.12

gradient_repeated_mixed_seconds=0.031468
gradient_repeated_mixed_queries_per_second=8135.29

mixed_unique_mixed_seconds=1.037080
mixed_unique_mixed_queries_per_second=7713.96

mixed_repeated_mixed_seconds=0.035390
mixed_repeated_mixed_queries_per_second=7233.64
```

## Main interpretation

- Warm cached calls are dramatically faster than cold calls.
- The old mixed throughput numbers make unique and repeated workloads look more similar than they really are, because they average cold and warm phases together.
- For value interpolation, warm cached throughput is roughly two orders of magnitude higher than cold throughput.
- Gradient and mixed calls also benefit strongly from caching, but they still do more work on the warm path than value-only interpolation.

## Structural refactor note

Commit: `dcf279c` (`Refactor interpolator cache flow`)

The first structural cleanup of `SoledgeHDG2DInterpolator` made the code easier to read and removed duplicated logic between `evaluate()` and `gradient()`, but it was slightly slower on the hot path. The likely reason is extra helper and dictionary-access overhead introduced by the cleaner structure.

That slowdown is acceptable for now because this file is now easier to optimize safely in the next pass.
