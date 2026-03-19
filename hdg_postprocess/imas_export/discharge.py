import gc
import json

import numpy as np

from .config import IMASExportMetadata, RectangularGrid2D, SolutionSnapshotSource
from .equilibrium import _equilibrium_metadata
from .evaluate import equilibrium_interpolators, evaluate_interpolators_on_grid, evaluate_variables_on_grid
from .ggd_geometry import populate_grid_reference_ggd_entry, populate_rectangular_grid_ggd_entry
from .summary import _build_ids_comment, extract_solution_summary_metadata


def write_discharge(
    entry,
    solutions,
    metadata: IMASExportMetadata,
    grid,
    *,
    include_summary=True,
    include_equilibrium=True,
    include_plasma_profiles=True,
    sort_by_time=False,
    time_getter=None,
):
    if not solutions:
        raise ValueError("Full-discharge export requires at least one solution snapshot.")
    if time_getter is None:
        raise ValueError("A time_getter callable must be provided for full-discharge export.")

    grids = _normalize_snapshot_grids(len(solutions), grid)
    timed_snapshots = _normalize_timed_snapshots(
        solutions,
        grids,
        sort_by_time=sort_by_time,
        time_getter=time_getter,
    )
    times = [time_value for time_value, _, _ in timed_snapshots]

    written = {}
    if include_summary:
        summary = build_discharge_summary_ids(timed_snapshots, metadata, times)
        entry.put(summary, metadata.occurrence)
        written["summary"] = _written_ids_info(metadata, times)
        _release_large_object(summary)

    grid_reference_paths = None
    if include_equilibrium and include_plasma_profiles:
        grid_reference_paths = _plasma_grid_reference_paths(metadata.occurrence, len(timed_snapshots))

    if include_plasma_profiles:
        plasma = build_discharge_plasma_profiles_ids(timed_snapshots, metadata)
        entry.put(plasma, metadata.occurrence)
        written["plasma_profiles"] = _written_ids_info(metadata, times)
        _release_large_object(plasma)

    if include_equilibrium:
        equilibrium = build_discharge_equilibrium_ids(
            timed_snapshots,
            metadata,
            grid_reference_paths=grid_reference_paths,
        )
        entry.put(equilibrium, metadata.occurrence)
        written["equilibrium"] = _written_ids_info(metadata, times)
        _release_large_object(equilibrium)

    return written


def build_discharge_summary_ids(timed_snapshots, metadata, times):
    import imas

    _, first_snapshot, _ = timed_snapshots[0]
    first_solution, owned = _load_snapshot(first_snapshot)

    summary = imas.IDSFactory().summary()
    summary.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    summary.description = metadata.description
    summary.time = np.asarray(times, dtype=float)
    summary.pulse = int(metadata.shot)

    if metadata.machine:
        summary.machine = metadata.machine

    try:
        extracted = extract_solution_summary_metadata(first_solution)
        extracted["export_shot"] = int(metadata.shot)
        extracted["export_run"] = int(metadata.run)
        extracted["export_occurrence"] = int(metadata.occurrence)
        extracted["effective_energy_transfer"] = float(metadata.effective_energy_transfer)
        extracted["snapshot_count"] = len(times)
        extracted["exported_times_s"] = [float(time_value) for time_value in times]

        summary.ids_properties.comment = _build_ids_comment(metadata, extracted)
        if metadata.comment:
            summary.tag.comment = metadata.comment

        summary.code.name = "SOLEDGE-HDG"
        summary.code.repository = "hdg_postprocess"
        testcase = extracted.get("testcase")
        if testcase is None:
            summary.code.description = "Exported full discharge from SOLEDGE-HDG by hdg_postprocess."
        else:
            summary.code.description = (
                f"Exported full discharge from SOLEDGE-HDG by hdg_postprocess (testcase {testcase})."
            )
        summary.code.parameters = json.dumps(extracted, sort_keys=True)
        summary.simulation.workflow = "time_dependent_discharge"

        puff_rate = extracted.get("puff_rate")
        if puff_rate is not None:
            summary.gas_injection_rates.total.value = np.full(len(times), float(puff_rate), dtype=float)
            summary.gas_injection_rates.total.source = "SOLEDGE-HDG physics/puff"
    finally:
        _release_snapshot(first_solution, owned)

    return summary


def build_discharge_plasma_profiles_ids(timed_snapshots, metadata: IMASExportMetadata):
    import imas

    first_time, first_snapshot, first_grid = timed_snapshots[0]
    first_solution, owned = _load_snapshot(first_snapshot)
    times = np.asarray([time_value for time_value, _, _ in timed_snapshots], dtype=float)

    plasma = imas.IDSFactory().plasma_profiles()
    plasma.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    plasma.time = times

    plasma.grid_ggd.resize(len(timed_snapshots))
    for grid_index, (time_value, _, grid) in enumerate(timed_snapshots, start=1):
        populate_rectangular_grid_ggd_entry(
            plasma.grid_ggd[grid_index - 1],
            grid,
            float(time_value),
            grid_name="rectangular_rz",
            grid_index=grid_index,
        )

    plasma.ggd.resize(len(timed_snapshots))
    for index, (time_value, snapshot, grid) in enumerate(timed_snapshots):
        solution, snapshot_owned = _load_snapshot(snapshot)
        try:
            _populate_plasma_ggd(solution, plasma.ggd[index], time_value, grid, grid_index=index + 1)
        finally:
            _release_snapshot(solution, snapshot_owned)

    try:
        if "Zeff" in first_solution.parameters["physics"]:
            plasma.global_quantities.z_eff_resistive = np.full(
                len(timed_snapshots),
                float(first_solution.parameters["physics"]["Zeff"]),
                dtype=float,
            )

        plasma.code.name = "SOLEDGE-HDG"
        plasma.code.repository = "hdg_postprocess"
        plasma.code.description = "Time-resolved plasma profiles exported from SOLEDGE-HDG by hdg_postprocess."
        plasma.code.parameters = json.dumps(
            _plasma_profiles_metadata(first_solution, first_grid, times),
            sort_keys=True,
        )
    finally:
        _release_snapshot(first_solution, owned)

    return plasma


def build_discharge_equilibrium_ids(timed_snapshots, metadata: IMASExportMetadata, *, grid_reference_paths=None):
    import imas

    first_time, first_snapshot, first_grid = timed_snapshots[0]
    first_solution, owned = _load_snapshot(first_snapshot)
    times = np.asarray([time_value for time_value, _, _ in timed_snapshots], dtype=float)

    eq = imas.IDSFactory().equilibrium()
    eq.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    eq.time = times

    eq.grids_ggd.resize(len(timed_snapshots))
    for grid_index, (time_value, _, grid) in enumerate(timed_snapshots, start=1):
        _populate_discharge_equilibrium_grid(
            eq.grids_ggd[grid_index - 1],
            grid,
            time_value=time_value,
            grid_index=grid_index,
            grid_reference_path=None if grid_reference_paths is None else grid_reference_paths[grid_index - 1],
        )

    eq.time_slice.resize(len(timed_snapshots))
    for index, (time_value, snapshot, grid) in enumerate(timed_snapshots):
        solution, snapshot_owned = _load_snapshot(snapshot)
        try:
            sampled_fields = _sample_equilibrium_fields(solution, grid)
            _fill_equilibrium_timeslice(eq.time_slice[index], sampled_fields, time_value=time_value, grid_index=index + 1)
        finally:
            _release_snapshot(solution, snapshot_owned)

    try:
        eq.code.name = "SOLEDGE-HDG"
        eq.code.repository = "hdg_postprocess"
        eq.code.description = "Time-resolved equilibrium exported from SOLEDGE-HDG by hdg_postprocess."

        eq_metadata = _equilibrium_metadata(first_solution, metadata, first_grid)
        eq_metadata["snapshot_count"] = len(timed_snapshots)
        eq_metadata["exported_times_s"] = times.tolist()
        eq_metadata["grid_count"] = len(timed_snapshots)
        eq_metadata["time_varying_grid"] = True
        eq_metadata["representation_note"] = (
            "Exported as a time-resolved discharge on rectangular cylindrical (R,Z) GGD meshes "
            "through equilibrium.grids_ggd[i]/time_slice[i].ggd."
        )
        if grid_reference_paths is not None:
            eq_metadata["ggd_grid_reference_note"] = (
                "equilibrium.grids_ggd[i] references the topology stored in plasma_profiles.grid_ggd[i] "
                "to avoid duplicate rectangular GGD geometry."
            )
        eq.code.parameters = json.dumps(eq_metadata, sort_keys=True)
    finally:
        _release_snapshot(first_solution, owned)

    return eq


def _normalize_snapshot_grids(snapshot_count, grid):
    if isinstance(grid, RectangularGrid2D):
        return [grid] * snapshot_count

    grids = list(grid)
    if len(grids) != snapshot_count:
        raise ValueError(
            "When exporting a full discharge with time-varying grids, the grid sequence "
            "must have the same length as the solution sequence."
        )
    for index, one_grid in enumerate(grids):
        if not isinstance(one_grid, RectangularGrid2D):
            raise TypeError(
                f"Grid {index} is not a RectangularGrid2D instance. "
                "The full-discharge exporter currently supports rectangular (R,Z) GGD grids."
            )
    return grids


def _normalize_timed_snapshots(snapshots, grids, *, sort_by_time, time_getter):
    timed_snapshots = []
    previous_time = None

    for index, (snapshot, grid) in enumerate(zip(snapshots, grids)):
        solution, owned = _load_snapshot(snapshot)
        try:
            time_value = time_getter(solution)
            if time_value is None:
                raise ValueError(
                    "Full-discharge export requires physically meaningful times for every snapshot. "
                    f"Snapshot {index} does not provide one."
                )
            time_value = float(time_value)
            if not sort_by_time and previous_time is not None and time_value < previous_time:
                raise ValueError(
                    "Full-discharge export expects snapshots to already be ordered in time. "
                    f"Snapshot {index} has time {time_value} s after {previous_time} s."
                )
            timed_snapshots.append((time_value, snapshot, grid))
            previous_time = time_value
        finally:
            _release_snapshot(solution, owned)

    if sort_by_time:
        timed_snapshots.sort(key=lambda item: item[0])
    return timed_snapshots


def _populate_plasma_ggd(solution, ggd, time_value, grid, *, grid_index):
    solution.assembly.full()
    solution.assembly.simple()

    ggd.time = float(time_value)

    r_grid, z_grid = grid.mesh()
    sampled = evaluate_variables_on_grid(
        solution,
        r_grid,
        z_grid,
        ["n", "te", "ti", "u", "nn", "psi"],
        locator=solution.mesh.geometry.element_locator,
        outside_value=np.nan,
    )

    _store_struct_field(ggd.electrons.density, sampled["n"], grid_index=grid_index)
    _store_struct_field(ggd.electrons.temperature, sampled["te"], grid_index=grid_index)

    ggd.ion.resize(1)
    ion = ggd.ion[0]
    ion.name = "D+"
    ion.z_ion = 1.0
    _store_struct_field(ion.temperature, sampled["ti"], grid_index=grid_index)
    ion.velocity.resize(1)
    ion.velocity[0].grid_index = int(grid_index)
    ion.velocity[0].grid_subset_index = 1
    ion.velocity[0].parallel = sampled["u"].reshape(-1)

    ggd.neutral.resize(1)
    neutral = ggd.neutral[0]
    neutral.name = "D"
    _store_struct_field(neutral.density, sampled["nn"], grid_index=grid_index)

    _store_struct_field(ggd.psi, sampled["psi"], grid_index=grid_index)


def _plasma_profiles_metadata(solution, grid, times):
    metadata_dict = {
        "grid_shape": [grid.nr, grid.nz],
        "grid_r_range_m": [grid.r_min, grid.r_max],
        "grid_z_range_m": [grid.z_min, grid.z_max],
        "ggd_grid_name": "rectangular_rz",
        "ggd_grid_subset": "All exported plasma fields currently live on the nodes subset.",
        "value_ordering": "Node values are flattened from meshgrid(indexing='ij') in C order, so R is the slow axis and Z the fast axis.",
        "outside_mesh_policy": "Values outside the HDG mesh are exported as NaN.",
        "model_note": "SOLEDGE-HDG currently uses shared plasma density and parallel velocity for electrons and the single ion species.",
        "density_storage_note": (
            "For the current single-ion SOLEDGE-HDG model, electrons.density is the authoritative density field. "
            "The redundant ion[0].density and n_i_total fields are intentionally left empty to reduce storage."
        ),
        "temperature_storage_note": (
            "For the current single-ion model, ion[0].temperature is populated and the redundant t_i_average field is left empty."
        ),
        "snapshot_count": len(times),
        "exported_times_s": list(times),
        "grid_count": len(times),
        "time_varying_grid": True,
    }
    if "Zeff" in solution.parameters["physics"]:
        metadata_dict["Zeff"] = float(solution.parameters["physics"]["Zeff"])
        metadata_dict["Zeff_storage_note"] = (
            "Spatially constant Zeff is exported through plasma_profiles.global_quantities.z_eff_resistive."
        )
    return metadata_dict


def _populate_discharge_equilibrium_grid(
    grids_ggd_entry,
    grid,
    *,
    time_value,
    grid_index,
    grid_reference_path,
):
    if grid_reference_path is None:
        populate_rectangular_grid_ggd_entry(
            grids_ggd_entry,
            grid,
            float(time_value),
            grid_name="rectangular_rz",
            grid_index=grid_index,
        )
        return

    populate_grid_reference_ggd_entry(
        grids_ggd_entry,
        time_value=float(time_value),
        path=str(grid_reference_path),
        grid_name="rectangular_rz",
        grid_index=grid_index,
    )


def _sample_equilibrium_fields(solution, grid):
    solution.assembly.full()
    solution.assembly.simple()
    solution.equilibrium.define_axis()

    r_grid, z_grid = grid.mesh()
    interpolators = equilibrium_interpolators(solution)

    eval_fields = {
        "psi": interpolators["psi"],
        "br": interpolators["br"],
        "bz": interpolators["bz"],
        "bphi": interpolators["bphi"],
    }
    if interpolators["jphi"] is not None:
        eval_fields["jphi"] = interpolators["jphi"]

    sampled = evaluate_interpolators_on_grid(
        eval_fields,
        r_grid=r_grid,
        z_grid=z_grid,
        locator=solution.mesh.geometry.element_locator,
        outside_value=np.nan,
    )

    axis = solution.summary.equilibrium.axis
    psi_values = sampled["psi"].reshape(-1)
    finite_psi = psi_values[np.isfinite(psi_values)]
    return {
        "psi": psi_values,
        "b_field_r": sampled["br"].reshape(-1),
        "b_field_z": sampled["bz"].reshape(-1),
        "b_field_phi": sampled["bphi"].reshape(-1),
        "j_phi": sampled["jphi"].reshape(-1) if "jphi" in sampled else None,
        "axis_r": float(axis.r) if axis.r is not None else None,
        "axis_z": float(axis.z) if axis.z is not None else None,
        "psi_axis": float(np.nanmin(finite_psi)) if finite_psi.size else None,
    }


def _fill_equilibrium_timeslice(ts, sampled_fields, *, time_value, grid_index):
    ts.time = float(time_value)
    ts.ggd.resize(1)
    ggd = ts.ggd[0]

    _store_equilibrium_field(ggd.psi, sampled_fields["psi"], grid_index=grid_index)
    _store_equilibrium_field(ggd.b_field_r, sampled_fields["b_field_r"], grid_index=grid_index)
    _store_equilibrium_field(ggd.b_field_z, sampled_fields["b_field_z"], grid_index=grid_index)
    _store_equilibrium_field(ggd.b_field_phi, sampled_fields["b_field_phi"], grid_index=grid_index)
    if sampled_fields["j_phi"] is not None:
        _store_equilibrium_field(ggd.j_phi, sampled_fields["j_phi"], grid_index=grid_index)

    if sampled_fields["axis_r"] is not None and sampled_fields["axis_z"] is not None:
        ts.global_quantities.magnetic_axis.r = sampled_fields["axis_r"]
        ts.global_quantities.magnetic_axis.z = sampled_fields["axis_z"]
    if sampled_fields["psi_axis"] is not None:
        ts.global_quantities.psi_axis = sampled_fields["psi_axis"]


def _plasma_grid_reference_paths(occurrence, grid_count):
    return [
        f"#plasma_profiles:{int(occurrence)}/grid_ggd({index})"
        for index in range(1, grid_count + 1)
    ]


def _written_ids_info(metadata, times):
    return {"occurrence": int(metadata.occurrence), "time_count": len(times)}


def _store_equilibrium_field(field_container, values, *, grid_index):
    field_container.resize(1)
    field_container[0].grid_index = int(grid_index)
    field_container[0].grid_subset_index = 1
    field_container[0].values = values


def _store_struct_field(field_container, values, *, grid_index):
    field_container.resize(1)
    field_container[0].grid_index = int(grid_index)
    field_container[0].grid_subset_index = 1
    field_container[0].values = values.reshape(-1)


def _load_snapshot(snapshot):
    if isinstance(snapshot, SolutionSnapshotSource):
        return snapshot.load(), True
    return snapshot, False


def _release_snapshot(snapshot, owned):
    if owned:
        del snapshot
        gc.collect()


def _release_large_object(value):
    del value
    gc.collect()
