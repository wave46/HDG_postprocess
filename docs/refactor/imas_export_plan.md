# IMAS Export Plan

This note defines the first IMAS export scope for `hdg_postprocess`.

The goal is not to cover every IMAS possibility at once. The goal is to produce a small, clear, shareable export path that lets other users open a SOLEDGE-HDG case from an IMAS `DBEntry` and use it for inspection or synthetic-diagnostic work.

## IMAS terms in this project

For this project, the most useful practical interpretation is:

- `DBEntry`
  - one opened IMAS database handle
  - tied to a backend and identifiers such as shot / run / occurrence
  - used to write or read several IDSs
- `IDS`
  - one schema-defined physics object stored inside the `DBEntry`
  - examples for this project: `equilibrium`, `plasma_profiles`, `summary`
- IDS fields
  - the nested schema-defined data inside one IDS
  - for example grids, profiles, global quantities, identifiers, and descriptive metadata

So the first exporter will:

1. open one `DBEntry`
2. fill a small set of IDSs
3. close the `DBEntry`

## V1 scope

The first export version should be intentionally narrow:

- backend: netCDF
- source object: one `HDGsolution`
- main IDSs:
  - `equilibrium`
  - `plasma_profiles`
  - `summary`
- spatial representation:
  - GGD first
  - start with a regular rectangular cylindrical `(R, Z)` mesh represented through GGD
- outside-mesh policy:
  - `NaN`

The first implementation should still prioritize ease of use for downstream readers, but the export contract should already align with the GGD branches that naturally carry species-resolved 2D data.

## Mandatory export metadata

The exporter should require a small metadata bundle from the user.

At minimum, the following fields should be mandatory in v1:

- `description`
  - short human-readable description of the simulation or case
- `shot`
- `run`
- `occurrence`
- `time`

Why these should be mandatory:

- they prevent creating anonymous or ambiguous database entries
- steady-state HDG outputs do not always carry a physically meaningful experiment time
- the same nominal shot may correspond to several exported simulation variants

Optional metadata can include:

- `comment`
- `case_name`
- `recycling_coefficient`
- impurity / cooling model label

`puff_rate` should be filled from the simulation data when it is already available in the loaded solution metadata.

In v1, puff rate and recycling should still be treated as run-level descriptive metadata rather than as wall-resolved data.

## V1 IDS content

### equilibrium

Planned content:

- poloidal flux
- magnetic field components
- toroidal / ohmic current where available

This IDS should be written as directly as possible from the current solution-equilibrium data already available in `HDGsolution`.
The long-term target is the `ggd` branch, even if a temporary rectangular `profiles_2d` representation is used during the transition.

### plasma_profiles

Planned content on a rectangular GGD mesh:

- electron density
- ion density
- parallel velocity
- electron temperature
- ion temperature
- neutral density
- optional derived fields that are already stable and easy to expose

Possible later additions:

- derived pressures
- Mach number
- source terms

Even in v1, the exporter should try to carry as much useful plasma information as practical, since the SOLEDGE-HDG model has a manageable number of primary fields.
Because the IMAS `plasma_profiles.ggd` branch already exposes `electrons`, `ion`, and `neutral`, it is the natural place for the first species-resolved 2D export.

### summary

Planned content:

- case description
- identifiers
- run-level metadata such as puff rate or recycling coefficient

This IDS is mainly meant to keep the export self-describing.

## Rectangular grid policy

The first plasma export should use a user-configurable regular cylindrical grid in `(R, Z)`.
This grid should be represented in GGD form from the start, even if its geometry is much simpler than the eventual HDG-native mesh description.

Suggested export controls:

- `r_min`, `r_max`, `nr`
- `z_min`, `z_max`, `nz`

The exporter should evaluate the HDG solution on that grid using the existing locator and interpolator path on the writer side.
Exporter code should reuse the normal `solution.sample.define_interpolators()` cache rather than building private interpolators for each IDS writer.

### Outside the mesh

Values outside the computational mesh should be exported as `NaN`.

That is safer than zero-filled defaults because:

- outside-domain points are clearly distinguishable
- downstream diagnostics do not silently treat non-physical regions as plasma vacuum with zero values
- it avoids accidental bias in averages or integrals

## Deferred items

These are intentionally out of scope for the first implementation:

- dual representation inside the same export flow
- full radiation IDS support
- wall-resolved recycling representation
- core-profile reductions derived from the 2D solution

These are all valid future directions, but they should not complicate the first usable exporter.

## Proposed implementation steps

1. create an IMAS smoke-test script
2. define a small export configuration / metadata object
3. implement `equilibrium` writer
4. implement rectangular-GGD `plasma_profiles` writer
5. implement minimal `summary` writer
6. add one readback example for users

## First success criterion

The first milestone is:

- one `HDGsolution`
- exported into one netCDF-backed IMAS `DBEntry`
- containing `equilibrium`, `plasma_profiles`, and `summary`
- with a simple readback example that plots one plasma field on `(R, Z)`

If that works cleanly, later extensions such as HDG-native GGD and radiation IDS export can be added on top of a stable base.
