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

- `plot_overview()`
- `plot_overview_physical()`
- `init_phys_variables()`
- `define_interpolators()`
- `calculate_variables_along_line()`
- `calculate_power_balance()`
- `calculate_boundary_summary()`
- `n()`
- `ti()`
- `te()`
- `nn()`
- `grad_ti()`
- `grad_ti_par()`

Properties and attributes:

- `mesh`
- `parameters`
- `atomic_parameters`
- `dnn_parameters`
- `boundary_summary`
- `solution_simple_phys`
- `gradient_simple_phys`
- `solution_glob_phys`
- `gradient_glob_phys`
- `phys_idx`

Preferred replacement API for new code:

- `views.simple.solution.physical`
- `views.simple.gradient.physical`
- `views.glob.solution.physical`
- `views.glob.gradient.physical`
- `summary.boundary.boundary_summary`

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
