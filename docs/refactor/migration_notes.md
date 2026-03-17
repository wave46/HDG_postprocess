# Migration Notes

## Current status

The refactor has moved past the heaviest compatibility bridge phase.

- Demo-covered loader entrypoints remain supported.
- `hdg_postprocess.api` is now a thin convenience loader that returns the structured `HDGsolution` directly.
- The preferred data-access style is now the structured container API:
  - `solution.views`
  - `solution.summary`
  - `solution.metadata`
  - `solution.additional_parameters`
  - `solution.atomic_rates`
  - `solution.interpolators`
- The preferred behavior-access style is now grouped facades on `HDGsolution`:
  - `solution.fields`
  - `solution.analysis`
  - `solution.sample`
  - `solution.plot`
  - `solution.assembly`
  - `solution.equilibrium`
  - `solution.sources`
  - `solution.neutrals`
  - `solution.turbulence`
  - `solution.pointwise`

## Recommended usage

For existing notebooks and analysis scripts:

- keep using `hdg_postprocess.formats.load_from_file`
- do not migrate working demo-style workflows unless there is a clear benefit

For new code:

- prefer `hdg_postprocess.api.load_solution()`
- prefer grouped access through facades such as `fields`, `analysis`, `sample`, `plot`, and `sources`
- prefer direct structured access such as `solution.views.simple.solution.physical`
- avoid introducing new code that depends on legacy split-property aliases; prefer paths like `solution.views.boundary_gauss.gradient.conservative`
- for point sampling, prefer the semantic pointwise subfacades:
  - `solution.pointwise.plasma`
  - `solution.pointwise.fields`
  - `solution.pointwise.gradients`
  - `solution.pointwise.fluxes`
  - `solution.pointwise.sources`

For existing notebooks and scripts that still use the old split-property names:

- update them gradually toward the structured container-and-facade model
- avoid adding new notebook cells that depend on removed flat wrappers such as old pointwise top-level methods

## Known limitations

- Some setup-heavy analyses such as power balance still require assigning atomic and neutral settings explicitly, just as in the legacy API.
- `raysect` is still used for production element location during interpolation.
- Notebook execution was not revalidated automatically in this cleanup phase because the local environment does not currently have `nbformat` / `nbconvert` installed.

## Next simplification targets

- clean `HDG_mesh.py` with the same package-first structure used for the solution API
- continue second-pass cleanup of `core/solution` and `routines`
- simplify facade internals where repeated orchestration is still visible
- expand tutor-style documentation around the structured API and migration path
- revisit a native compiled locator after the broader refactor is complete
