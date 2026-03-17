# Interpolator Benchmark Baseline

This note keeps the current benchmark baseline for
[benchmark_interpolators.py](/home/ikudashev/Documents/Github/HDG_postprocess/scripts/benchmark_interpolators.py)
and serves as the comparison point for the next optimization pass in
[interpolators.py](/home/ikudashev/Documents/Github/HDG_postprocess/hdg_postprocess/routines/interpolators.py).

## Command

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tofu-cherab
python scripts/benchmark_interpolators.py
```

## Workload definition

- `*_cold`: first pass over a cleared cache
- `*_warm`: immediate second pass over the same cached points
- `*_mixed`: the multi-pass average over `repeat` traversals after a cache clear

The benchmark uses the real solution-loading and `define_interpolators()` path, so locator work is included.

## Current baseline

Scenario: `legacy_first`

Current compact defaults:

```bash
python scripts/benchmark_interpolators.py
```

### Cold passes

```text
value_unique_cold_seconds=0.044717
value_unique_cold_queries_per_second=11181.44

value_repeated_cold_seconds=0.002400
value_repeated_cold_queries_per_second=13334.17

gradient_unique_cold_seconds=0.040530
gradient_unique_cold_queries_per_second=12336.56

gradient_repeated_cold_seconds=0.002612
gradient_repeated_cold_queries_per_second=12251.96

mixed_unique_cold_seconds=0.041347
mixed_unique_cold_queries_per_second=12092.66

mixed_repeated_cold_seconds=0.002432
mixed_repeated_cold_queries_per_second=13156.52
```

### Warm passes

```text
value_unique_warm_seconds=0.001150
value_unique_warm_queries_per_second=434616.34

value_repeated_warm_seconds=0.000071
value_repeated_warm_queries_per_second=453148.98

gradient_unique_warm_seconds=0.002223
gradient_unique_warm_queries_per_second=224942.54

gradient_repeated_warm_seconds=0.000193
gradient_repeated_warm_queries_per_second=166134.44

mixed_unique_warm_seconds=0.003087
mixed_unique_warm_queries_per_second=161945.16

mixed_repeated_warm_seconds=0.000279
mixed_repeated_warm_queries_per_second=114718.40
```

### Mixed multi-pass results

```text
value_unique_mixed_seconds=0.042273
value_unique_mixed_queries_per_second=23655.88

value_repeated_mixed_seconds=0.002714
value_repeated_mixed_queries_per_second=23577.24

gradient_unique_mixed_seconds=0.043149
gradient_unique_mixed_queries_per_second=23175.63

gradient_repeated_mixed_seconds=0.003078
gradient_repeated_mixed_queries_per_second=20791.96

mixed_unique_mixed_seconds=0.043420
mixed_unique_mixed_queries_per_second=23030.97

mixed_repeated_mixed_seconds=0.002654
mixed_repeated_mixed_queries_per_second=24116.61
```

## Main interpretation

- Warm cached calls are dramatically faster than cold calls.
- The old mixed throughput numbers make unique and repeated workloads look more similar than they really are, because they average cold and warm phases together.
- For value interpolation, warm cached throughput is roughly two orders of magnitude higher than cold throughput.
- Gradient and mixed calls also benefit strongly from caching, but they still do more work on the warm path than value-only interpolation.
- After the structural cleanup, the safe optimization passes recovered the earlier mixed-path regression.
- The first compiled triangle-scalar kernel pass then improved cold-path throughput by roughly a factor of 4 for the compact benchmark defaults.

## Refactor note

- `dcf279c` (`Refactor interpolator cache flow`) made the class cleaner but slightly slower.
- `58a770b` (`Optimize interpolator fast path`) recovered most of the hot-path regression and improved warm cached calls noticeably.
- The current benchmark numbers also include the follow-up allocation-reduction pass in `xieta_element()`, `xieta_element_precise()`, and `_compute_shape_data()`.
- The current benchmark numbers also include an optional Cython fast path in `hdg_postprocess.routines._interpolators_fast` for the triangle-scalar orthogonal polynomial routines.

## Evolution summary

- Original pure-Python baseline on the older large workload:
  - mixed unique throughput was roughly `7.7k` to `9.2k q/s`, depending on workload
- Structural cleanup:
  - improved readability
  - introduced a small regression that later Python cleanup recovered
- Current compact baseline after Python cleanup plus the first Cython pass:
  - `value_unique_mixed`: `23.7k q/s`
  - `gradient_unique_mixed`: `23.2k q/s`
  - `mixed_unique_mixed`: `23.0k q/s`
  - cold miss-path throughput is now roughly `4x` higher than the compact pure-Python baseline

## Likely remaining bottlenecks

With the compiled polynomial path in place, the cold-path cost is now dominated more clearly by:

- `xieta_element_precise()`
- `_compute_shape_data()`
- the remaining NumPy dot/Jacobian work around the compiled polynomial kernels

The next best compiled target is probably the local-coordinate solve in `xieta_element_precise()`. The profile after the first Cython pass shows that the polynomial routines are no longer the dominant cost center, which is exactly the shift we wanted to see.
