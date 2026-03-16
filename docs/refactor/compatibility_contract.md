# Demo Compatibility Contract

This document records the backward-compatibility surface that the refactor must preserve.
The contract is derived from the notebooks under `demos/` and is intentionally narrower
than "every internal attribute currently present on the classes".

## Public entrypoints

The following public entrypoints are exercised directly in the demos and must remain
compatible throughout the refactor:

- `hdg_postprocess.formats.load_from_file.load_HDG_solution_from_file()`
- `hdg_postprocess.formats.load_from_file.load_HDG_mesh_from_file()`
- `hdg_postprocess.HDG_solution.HDGsolution`
- `hdg_postprocess.HDG_mesh.HDGmesh`

## `HDGsolution` demo-covered surface

Methods:

- `plot.overview()`
- `plot.physical_overview()`
- `fields.initialize_physical()`
- `sample.define_interpolators()`
- `sample.line()`
- `analysis.power_balance()`
- `analysis.boundary_summary()`
- `analysis.wall_profile()`
- `pointwise.plasma.n()`
- `pointwise.plasma.ti()`
- `pointwise.plasma.te()`
- `pointwise.plasma.nn()`
- `pointwise.gradients.ti()`
- `pointwise.gradients.ti_parallel()`

Properties and attributes:

- `mesh`
- `parameters`
- `additional_parameters`

Preferred replacement API for new code:

- `views.simple.solution.physical`
- `views.simple.gradient.physical`
- `views.glob.solution.physical`
- `views.glob.gradient.physical`
- `summary.boundary.profile`

Required workflows:

- loading legacy partitioned runs with an external mesh
- loading newer single-file runs with an embedded mesh
- initializing physical variables from conservative values
- creating interpolators and sampling along a line
- evaluating power balance and wall summaries when atomic settings are provided

## `HDGmesh` demo-covered surface

Methods:

- `plot_raw_meshes()`
- `plot_full_mesh()`
- `recombine_full_mesh()`
- `create_connectivity_big()`
- `make_element_number_funtion()`
- `calculate_gauss_volumes()`
- `find_adjacent_elements()`

Properties and attributes:

- `element_number`
- `reference_element`
- `vertices_glob`
- `connectivity_glob`
- `connectivity_big`
- `volumes_gauss`
- `mesh_extent`

Required workflows:

- loading a partitioned external mesh
- recombining the full mesh
- locating an element for a given point
- computing Gauss-point volumes after a reference element is assigned

## Dataset families covered by the demos

- legacy flat, partitioned solution files with external mesh:
  - `first`
  - `detached`
  - `high_recycling`
  - `sheath_limited`
- newer grouped single-file solutions with embedded mesh:
  - `k_equation_new_version_of_the_code`
  - `power_balance_boundary_checks`

## Baseline categories required before refactoring

- loading behavior and metadata
- mesh recombination outputs
- physical-variable initialization outputs
- power-balance summaries
- boundary summaries
- interpolated point values, including midplane points
- sampled line profiles, including `k` where present
