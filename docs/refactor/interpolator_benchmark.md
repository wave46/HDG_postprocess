# Interpolator Benchmark Baseline

This note captures the pre-refactor timing baseline for
[benchmark_interpolators.py](/home/ikudashev/Documents/Github/HDG_postprocess/scripts/benchmark_interpolators.py).
It is intended as the comparison point for the upcoming structural cleanup and any later optimization pass in
[interpolators.py](/home/ikudashev/Documents/Github/HDG_postprocess/hdg_postprocess/routines/interpolators.py).

## Command

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tofu-cherab
python scripts/benchmark_interpolators.py --scenario legacy_first --repeat 4 --unique-points 2000 --repeated-points 64
```

## Workload definition

- `value_unique`: scalar interpolation at 2000 unique interior points
- `value_repeated`: scalar interpolation over a 64-point repeated set
- `gradient_unique`: gradient interpolation at 2000 unique interior points
- `gradient_repeated`: gradient interpolation over a 64-point repeated set
- `mixed_unique`: value and gradient evaluated at the same 2000 unique points
- `mixed_repeated`: value and gradient evaluated over the same 64-point repeated set

The benchmark clears the shared interpolator point cache before each workload, so each timing starts from a cold cache and is comparable across modes.

## Baseline results

Scenario: `legacy_first`

```text
value_unique_seconds=0.841197
value_unique_queries=8000
value_unique_queries_per_second=9510.26

value_repeated_seconds=0.027640
value_repeated_queries=256
value_repeated_queries_per_second=9261.89

gradient_unique_seconds=0.872835
gradient_unique_queries=8000
gradient_unique_queries_per_second=9165.54

gradient_repeated_seconds=0.027536
gradient_repeated_queries=256
gradient_repeated_queries_per_second=9297.05

mixed_unique_seconds=0.880760
mixed_unique_queries=8000
mixed_unique_queries_per_second=9083.07

mixed_repeated_seconds=0.029209
mixed_repeated_queries=256
mixed_repeated_queries_per_second=8764.43
```

## Notes

- These measurements include the current exact-point cache logic.
- Because the cache is cleared before each workload, the repeated-point timings here are not a fully warm-cache best case; they represent the current cold-start behavior of each workload family.
- Locator time is part of the measured interpolation path because the benchmark uses the real `define_interpolators()` setup and live element lookup.
