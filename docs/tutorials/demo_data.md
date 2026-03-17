# Demo Data

Some demos and benchmark workflows need auxiliary local files that are not part of the minimal public package layout.

## What counts as demo data

The main auxiliary bundles are:

- reference-element `.mat` files
- atomic-rate and cooling-factor `.npy` tables

These files are used by the higher-level setup helpers, notebooks, and regression/benchmark tooling.

## When you need it

You typically need reference-element data for:

- Gauss-volume assembly
- boundary-Gauss assembly
- interpolator construction
- point and line sampling

You typically need atomic data for:

- power-balance workflows
- ionization / recombination / charge-exchange source terms
- optional cooling-factor calculations

## Expected local layout

The demos assume a simple local structure such as:

```text
data/
  reference_elements/
    reference_triangle_P4.mat
    reference_triangle_P8.mat
  atomic/
    alpha_iz.npy
    alpha_rec_2.1.8JH.npy
    alpha_energy_iz.npy
    alpha_energy_rec.npy
    LZ_Nitrogen_adas_fit_te_2e-1_4e3.npy
    LZ_Tungsten_adas_fit_te_2e0_4e4.npy
```

The path does not need to be exactly `data/`, but the setup helpers expect you to pass the location explicitly.

## Minimal setup examples

Reference element only:

```python
from hdg_postprocess.api import load_reference_element

solution.mesh.metadata.reference_element = load_reference_element(
    "data/reference_elements/reference_triangle_P8.mat"
)
```

Reference element plus atomic and neutral-diffusion setup:

```python
from hdg_postprocess.api import configure_solution_setup

configure_solution_setup(
    solution,
    reference_element="data/reference_elements/reference_triangle_P8.mat",
    radiation_model="nitrogen_cooling",
    atomic_data_dir="data/atomic",
    neutral_diffusion=True,
)
```

## Availability

These auxiliary files are available on request when you need to run the full demos, baseline regeneration, or benchmark workflows. The core package structure does not assume they are public repository contents.

## Related docs

- [Solution Quickstart](solution_quickstart.md)
- [Point Sampling](point_sampling.md)
- [Power Balance](power_balance.md)
