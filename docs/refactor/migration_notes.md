# Migration Notes

## Current status

The refactor is still in the compatibility-first phase.

- Demo-covered legacy entrypoints remain supported.
- `hdg_postprocess.api` provides the new additive API for new code.
- The new API currently delegates to the legacy implementation internally.

## Recommended usage

For existing notebooks and analysis scripts:

- keep using `hdg_postprocess.formats.load_from_file`
- do not migrate working demo-style workflows unless there is a clear benefit

For new code:

- prefer `hdg_postprocess.api.load_solution()`
- prefer grouped access through `fields`, `analysis`, `sample`, and `plot`

## Known limitations

- The modern API is not yet the primary internal implementation.
- Some setup-heavy analyses such as power balance still require assigning atomic and neutral settings explicitly, just as in the legacy API.
- `raysect` is still used for production element location during interpolation.

## Next simplification targets

- continue extracting logic from `HDG_solution.py`
- reduce duplicated orchestration across simple, full, gauss, and boundary representations
- separate pure computation from plotting and cache management
- clarify setup flows for atomic, neutral, and power-balance calculations
- revisit a native compiled locator after the broader refactor is complete
