import numpy as np
import pytest

imas = pytest.importorskip("imas")

from hdg_postprocess.api import load_reference_element, load_solution
from hdg_postprocess.imas_export import (
    IMASExportMetadata,
    RectangularGrid2D,
    equilibrium_slice,
    load_ids,
    plasma_slice,
    solution_time_seconds,
    write_discharge_imas_netcdf,
    write_imas_netcdf,
)
from hdg_postprocess.imas_export.equilibrium import sample_equilibrium_fields
from hdg_postprocess.imas_export.plasma_profiles import sample_plasma_fields

from helpers import require_scenario_data, scenario_map


def _load_solution_from_scenario(cfg):
    solution = load_solution(
        cfg["solution_path"],
        cfg["solution_base"],
        cfg.get("mesh_path"),
        cfg.get("mesh_base"),
        cfg["n_partitions"],
    )
    solution.mesh.metadata.reference_element = load_reference_element(cfg["reference_element"])
    return solution


def _set_solution_time(solution, value_seconds):
    solution.parameters.setdefault("switches", {})["steady"] = False
    solution.parameters.setdefault("time", {})["Current_time"] = float(value_seconds)
    solution.parameters.setdefault("adimensionalization", {})["time_scale"] = 1.0


def test_single_snapshot_imas_roundtrip_matches_sampled_fields(manifest_path, tmp_path):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["embedded_k_model"]
    require_scenario_data(cfg)

    solution = _load_solution_from_scenario(cfg)
    grid = RectangularGrid2D.from_solution_bounds(solution, dr=0.05, dz=0.05)
    metadata = IMASExportMetadata(
        description="roundtrip single snapshot",
        shot=1,
        run=1,
        time=0.0,
        effective_energy_transfer=0.0,
        occurrence=0,
    )
    db_path = tmp_path / "single_snapshot_roundtrip.nc"

    write_imas_netcdf(solution, db_path, metadata, grid)

    _, equilibrium, plasma = load_ids(db_path, occurrence=0)
    actual_eq = equilibrium_slice(equilibrium, plasma, time_index=0)
    actual_plasma = plasma_slice(plasma, time_index=0)

    expected_eq = sample_equilibrium_fields(solution, grid)
    expected_plasma = sample_plasma_fields(solution, grid)

    assert np.allclose(actual_eq["psi"], expected_eq["psi"].reshape(grid.nr, grid.nz), equal_nan=True)
    assert np.allclose(actual_eq["br"], expected_eq["b_field_r"].reshape(grid.nr, grid.nz), equal_nan=True)
    assert np.allclose(actual_eq["bz"], expected_eq["b_field_z"].reshape(grid.nr, grid.nz), equal_nan=True)
    assert np.allclose(actual_eq["bphi"], expected_eq["b_field_phi"].reshape(grid.nr, grid.nz), equal_nan=True)
    if expected_eq["j_phi"] is not None:
        assert actual_eq["jphi"] is not None
        assert np.allclose(actual_eq["jphi"], expected_eq["j_phi"].reshape(grid.nr, grid.nz), equal_nan=True)

    assert np.allclose(actual_plasma["ne"], expected_plasma["n"], equal_nan=True)
    assert np.allclose(actual_plasma["te"], expected_plasma["te"], equal_nan=True)
    assert np.allclose(actual_plasma["ti"], expected_plasma["ti"], equal_nan=True)
    assert np.allclose(actual_plasma["u_par"], expected_plasma["u"], equal_nan=True)
    assert np.allclose(actual_plasma["nn"], expected_plasma["nn"], equal_nan=True)
    assert np.allclose(actual_plasma["psi"], expected_plasma["psi"], equal_nan=True)


def test_discharge_roundtrip_resolves_reused_grid_paths(manifest_path, tmp_path):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["embedded_k_model"]
    require_scenario_data(cfg)

    solutions = []
    for time_value in (0.0, 1.0, 2.0, 3.0):
        solution = _load_solution_from_scenario(cfg)
        _set_solution_time(solution, time_value)
        solutions.append(solution)

    grid = RectangularGrid2D.from_solution_bounds(solutions[0], dr=0.06, dz=0.06)
    metadata = IMASExportMetadata(
        description="roundtrip discharge",
        shot=2,
        run=2,
        time=0.0,
        effective_energy_transfer=0.0,
        occurrence=0,
    )
    db_path = tmp_path / "discharge_roundtrip.nc"

    write_discharge_imas_netcdf(
        solutions,
        db_path,
        metadata,
        grid,
        time_getter=solution_time_seconds,
        sample_workers=1,
    )

    _, equilibrium, plasma = load_ids(db_path, occurrence=0)
    assert str(plasma.grid_ggd[1].path) == "#plasma_profiles:0/grid_ggd(1)"
    assert str(plasma.grid_ggd[3].path) == "#plasma_profiles:0/grid_ggd(1)"

    actual_eq = equilibrium_slice(equilibrium, plasma, time_index=3)
    actual_plasma = plasma_slice(plasma, time_index=3)

    expected_eq = sample_equilibrium_fields(solutions[3], grid)
    expected_plasma = sample_plasma_fields(solutions[3], grid)

    assert np.allclose(actual_eq["psi"], expected_eq["psi"].reshape(grid.nr, grid.nz), equal_nan=True)
    assert np.allclose(actual_eq["br"], expected_eq["b_field_r"].reshape(grid.nr, grid.nz), equal_nan=True)
    assert np.allclose(actual_eq["bz"], expected_eq["b_field_z"].reshape(grid.nr, grid.nz), equal_nan=True)
    assert np.allclose(actual_eq["bphi"], expected_eq["b_field_phi"].reshape(grid.nr, grid.nz), equal_nan=True)
    assert np.allclose(actual_plasma["ne"], expected_plasma["n"], equal_nan=True)
    assert np.allclose(actual_plasma["te"], expected_plasma["te"], equal_nan=True)
    assert np.allclose(actual_plasma["ti"], expected_plasma["ti"], equal_nan=True)
    assert np.allclose(actual_plasma["u_par"], expected_plasma["u"], equal_nan=True)
    assert np.allclose(actual_plasma["nn"], expected_plasma["nn"], equal_nan=True)
    assert np.allclose(actual_plasma["psi"], expected_plasma["psi"], equal_nan=True)
