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
value_unique_cold_seconds=0.054540
value_unique_cold_queries_per_second=9167.63

value_repeated_cold_seconds=0.002949
value_repeated_cold_queries_per_second=10850.30

gradient_unique_cold_seconds=0.052103
gradient_unique_cold_queries_per_second=9596.41

gradient_repeated_cold_seconds=0.003386
gradient_repeated_cold_queries_per_second=9449.86

mixed_unique_cold_seconds=0.053358
mixed_unique_cold_queries_per_second=9370.65

mixed_repeated_cold_seconds=0.003355
mixed_repeated_cold_queries_per_second=9538.74
```

### Warm passes

```text
value_unique_warm_seconds=0.001469
value_unique_warm_queries_per_second=340483.03

value_repeated_warm_seconds=0.000087
value_repeated_warm_queries_per_second=366901.11

gradient_unique_warm_seconds=0.002467
gradient_unique_warm_queries_per_second=202711.79

gradient_repeated_warm_seconds=0.000230
gradient_repeated_warm_queries_per_second=139312.14

mixed_unique_warm_seconds=0.003722
mixed_unique_warm_queries_per_second=134331.33

mixed_repeated_warm_seconds=0.000352
mixed_repeated_warm_queries_per_second=90882.76
```

### Mixed multi-pass results

```text
value_unique_mixed_seconds=0.049582
value_unique_mixed_queries_per_second=20168.49

value_repeated_mixed_seconds=0.003152
value_repeated_mixed_queries_per_second=20304.38

gradient_unique_mixed_seconds=0.054168
gradient_unique_mixed_queries_per_second=18460.94

gradient_repeated_mixed_seconds=0.003683
gradient_repeated_mixed_queries_per_second=17378.45

mixed_unique_mixed_seconds=0.059342
mixed_unique_mixed_queries_per_second=16851.55

mixed_repeated_mixed_seconds=0.003773
mixed_repeated_mixed_queries_per_second=16964.69
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

## Short benchmark history

- Original pure-Python baseline, large workload (`repeat=4`, `unique_points=2000`, `repeated_points=64`):
  - `value_unique_mixed`: about `9.2k q/s`
  - `gradient_unique_mixed`: about `9.0k q/s`
  - `mixed_unique_mixed`: about `7.7k q/s`
- Structural cleanup only:
  - code became cleaner
  - hot-path performance regressed slightly
- Python fast-path cleanup and allocation reduction:
  - mostly recovered the regression
  - warm cached calls improved noticeably
- First Cython triangle-scalar kernel pass with compact defaults (`repeat=2`, `unique_points=500`, `repeated_points=32`):
  - `value_unique_mixed`: about `20.2k q/s`
  - `gradient_unique_mixed`: about `18.5k q/s`
  - `mixed_unique_mixed`: about `16.9k q/s`
  - cold miss-path throughput improved by roughly a factor of `4` compared with the compact pure-Python baseline

## Likely remaining bottlenecks

With the compiled polynomial path in place, the cold-path cost is now dominated more clearly by:

- `xieta_element_precise()`
- `_compute_shape_data()`
- the remaining NumPy dot/Jacobian work around the compiled polynomial kernels

The next best compiled target is probably the local-coordinate solve in `xieta_element_precise()`. The profile after the first Cython pass shows that the polynomial routines are no longer the dominant cost center, which is exactly the shift we wanted to see.
