# Power Balance

This guide shows the typical setup needed before calling the integrated power-balance workflow.

## Why extra setup is needed

Power-balance terms depend on more than the raw HDG solution:

- a reference element is needed for Gauss and boundary-Gauss assembly
- atomic tables are needed for ionization, recombination, charge-exchange, and optional cooling terms
- neutral-diffusion settings are needed when neutral-derived terms are involved

The helper API is meant to collect that setup in one place.

## Typical workflow

```python
from hdg_postprocess.api import configure_solution_setup, load_solution

solution = load_solution(solution_path, solution_base, mesh_path, mesh_base, n_partitions)

configure_solution_setup(
    solution,
    reference_element=reference_element_path,
    radiation_model="nitrogen_cooling",
    atomic_data_dir="path/to/atomic",
    neutral_diffusion=True,
)
```

This does three things:

- loads and attaches the reference element to `solution.mesh.metadata.reference_element`
- loads the requested atomic parameter bundle from the explicit local data directory
- fills the default neutral-diffusion parameter bundle and adimensionalized copies

## Evaluate integrated power balance

```python
power = solution.analysis.power_balance()
```

This computes and caches the global integrated sources and sinks, including the atomic and optional cooling terms implied by the configured setup.

Typical entries include totals such as:

```python
power["total_gain"]
power["total_loss"]
power["total"]
```

## Old versus new setup style

Instead of manually loading `.mat` and `.npy` files in the notebook and then calling:

```python
solution.additional_parameters.set_atomic(...)
solution.additional_parameters.set_neutral_diffusion(...)
```

the preferred modern style is to collect that setup through `configure_solution_setup(...)`.

## Related docs

- [Solution Quickstart](solution_quickstart.md)
- [Point Sampling](point_sampling.md)
- [Boundary Summary](boundary_summary.md)
- [Setup Helpers](setup_helpers.md)
- demo notebook [hdg_solution_power_balance.ipynb](../../demos/hdg_solution_power_balance.ipynb)
