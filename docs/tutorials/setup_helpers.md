# Setup Helpers

Some workflows need a bit more than just loading a mesh or solution. This guide collects the small helper utilities used to attach reference elements and configure atomic, neutral-diffusion, and turbulence parameter bundles.

## What these helpers are for

The main public setup helpers are:

- `load_reference_element(path)`
- `make_atomic_parameters(radiation_model, data_dir=...)`
- `make_neutral_diffusion_parameters()`
- `make_turbulence_parameters()`
- `configure_solution_setup(...)`

They exist to keep notebooks and scripts from repeating long `.mat` and `.npy` loading blocks.

## When you need them

You typically need a reference element for:

- Gauss-volume assembly
- boundary-Gauss assembly
- interpolator construction
- point and line sampling

You typically need atomic data for:

- power-balance workflows
- ionization / recombination / charge-exchange source terms
- optional cooling-factor calculations

You typically need neutral-diffusion and turbulence parameters for:

- neutral transport-derived quantities such as `dnn` and `mfp`
- turbulence-derived quantities such as `dk`

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

The path does not need to be exactly `data/`, but the helpers expect you to pass the location explicitly.

## Reference element only

```python
from hdg_postprocess.api import load_reference_element

solution.mesh.metadata.reference_element = load_reference_element(
    "data/reference_elements/reference_triangle_P8.mat"
)
```

This is the minimum setup for interpolator construction and sampling workflows.

## Build parameter bundles explicitly

```python
from hdg_postprocess.api import (
    make_atomic_parameters,
    make_neutral_diffusion_parameters,
    make_turbulence_parameters,
)

atomic = make_atomic_parameters("nitrogen_cooling", data_dir="data/atomic")
dnn = make_neutral_diffusion_parameters()
dk = make_turbulence_parameters()
```

Use these lower-level helpers when you want to keep setup steps explicit in a script or override one of the returned dictionaries before attaching it to a solution.

## Configure a solution in one step

```python
from hdg_postprocess.api import configure_solution_setup

configure_solution_setup(
    solution,
    reference_element="data/reference_elements/reference_triangle_P8.mat",
    radiation_model="nitrogen_cooling",
    atomic_data_dir="data/atomic",
    neutral_diffusion=True,
    turbulence=False,
)
```

This helper:

- loads and attaches the reference element
- builds and attaches the requested atomic bundle
- fills and adimensionalizes the default neutral-diffusion bundle when requested
- fills and adimensionalizes the default turbulence bundle when requested

## Availability

These auxiliary files are available on request when you need to run the full demos, baseline regeneration, or benchmark workflows. The core package structure does not assume they are public repository contents.

## Related docs

- [Solution Quickstart](solution_quickstart.md)
- [Point Sampling](point_sampling.md)
- [Power Balance](power_balance.md)
- [Boundary Summary](boundary_summary.md)
