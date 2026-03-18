import json

import numpy as np

from .config import IMASExportMetadata, RectangularGrid2D
from .equilibrium import _equilibrium_metadata
from .evaluate import evaluate_variables_on_grid, equilibrium_interpolators, evaluate_interpolators_on_grid
from .ggd_geometry import populate_rectangular_grid_ggd, populate_rectangular_grid_ggd_array
from .summary import _build_ids_comment, extract_solution_summary_metadata


def _normalize_snapshot_times(solutions, *, sort_by_time, time_getter):
    timed = []
    for index, solution in enumerate(solutions):
        time_value = time_getter(solution)
        if time_value is None:
            raise ValueError(
                "Full-discharge export requires physically meaningful times for every snapshot. "
                f"Snapshot {index} does not provide one."
            )
        timed.append((float(time_value), solution))
    if sort_by_time:
        timed.sort(key=lambda item: item[0])
    return timed


def _build_discharge_summary_ids(solutions, metadata, times):
    import imas

    first_solution = solutions[0]
    summary = imas.IDSFactory().summary()
    summary.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    summary.description = metadata.description
    summary.time = np.asarray(times, dtype=float)
    summary.pulse = int(metadata.shot)

    if metadata.machine:
        summary.machine = metadata.machine

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

    return summary


def _populate_equilibrium_timeslice(solution, ts, time_value, grid, metadata):
    solution.assembly.full()
    solution.assembly.simple()
    solution.equilibrium.define_axis()

    ts.time = float(time_value)
    ts.ggd.resize(1)
    ggd = ts.ggd[0]

    r_grid, z_grid = grid.mesh()
    interpolators = equilibrium_interpolators(solution)
    locator = solution.mesh.geometry.element_locator
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
        locator=locator,
        outside_value=np.nan,
    )
    psi_values = sampled["psi"].reshape(-1)
    br_values = sampled["br"].reshape(-1)
    bz_values = sampled["bz"].reshape(-1)
    bphi_values = sampled["bphi"].reshape(-1)
    r_values = r_grid.reshape(-1)
    z_values = z_grid.reshape(-1)

    def _store_ggd_field(field_name, values):
        field = getattr(ggd, field_name)
        field.resize(1)
        field[0].grid_index = 1
        field[0].grid_subset_index = 1
        field[0].values = values

    _store_ggd_field("r", r_values)
    _store_ggd_field("z", z_values)
    _store_ggd_field("psi", psi_values)
    _store_ggd_field("b_field_r", br_values)
    _store_ggd_field("b_field_z", bz_values)
    _store_ggd_field("b_field_phi", bphi_values)
    if interpolators["jphi"] is not None:
        _store_ggd_field("j_phi", sampled["jphi"].reshape(-1))

    axis = solution.summary.equilibrium.axis
    if axis.r is not None and axis.z is not None:
        ts.global_quantities.magnetic_axis.r = float(axis.r)
        ts.global_quantities.magnetic_axis.z = float(axis.z)

    finite_psi = psi_values[np.isfinite(psi_values)]
    if finite_psi.size:
        ts.global_quantities.psi_axis = float(np.nanmin(finite_psi))


def build_discharge_equilibrium_ids(timed_solutions, metadata: IMASExportMetadata, grid: RectangularGrid2D):
    import imas

    first_time, first_solution = timed_solutions[0]
    times = np.asarray([time_value for time_value, _ in timed_solutions], dtype=float)

    eq = imas.IDSFactory().equilibrium()
    eq.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    eq.time = times
    populate_rectangular_grid_ggd(eq, grid, float(first_time))
    eq.time_slice.resize(len(timed_solutions))

    for index, (time_value, solution) in enumerate(timed_solutions):
        _populate_equilibrium_timeslice(solution, eq.time_slice[index], time_value, grid, metadata)

    eq.code.name = "SOLEDGE-HDG"
    eq.code.repository = "hdg_postprocess"
    eq.code.description = "Time-resolved equilibrium exported from SOLEDGE-HDG by hdg_postprocess."
    eq_metadata = _equilibrium_metadata(first_solution, metadata, grid)
    eq_metadata["snapshot_count"] = len(timed_solutions)
    eq_metadata["exported_times_s"] = times.tolist()
    eq_metadata["representation_note"] = (
        "Exported as a time-resolved discharge on a rectangular cylindrical (R,Z) mesh "
        "through equilibrium.grids_ggd/time_slice[i].ggd."
    )
    eq.code.parameters = json.dumps(eq_metadata, sort_keys=True)
    return eq


def _store_struct_field(field_container, values):
    field_container.resize(1)
    field_container[0].grid_index = 1
    field_container[0].grid_subset_index = 1
    field_container[0].values = values.reshape(-1)


def _populate_plasma_ggd(solution, ggd, time_value, grid):
    solution.assembly.full()
    solution.assembly.simple()

    ggd.time = float(time_value)

    r_grid, z_grid = grid.mesh()
    locator = solution.mesh.geometry.element_locator
    sampled = evaluate_variables_on_grid(
        solution,
        r_grid,
        z_grid,
        ["n", "te", "ti", "u", "nn", "psi"],
        locator=locator,
        outside_value=np.nan,
    )

    _store_struct_field(ggd.electrons.density, sampled["n"])
    _store_struct_field(ggd.electrons.temperature, sampled["te"])

    ggd.ion.resize(1)
    ion = ggd.ion[0]
    ion.name = "D+"
    ion.z_ion = 1.0
    _store_struct_field(ion.density, sampled["n"])
    _store_struct_field(ion.temperature, sampled["ti"])
    ion.velocity.resize(1)
    ion.velocity[0].grid_index = 1
    ion.velocity[0].grid_subset_index = 1
    ion.velocity[0].parallel = sampled["u"].reshape(-1)

    ggd.neutral.resize(1)
    neutral = ggd.neutral[0]
    neutral.name = "D"
    _store_struct_field(neutral.density, sampled["nn"])

    _store_struct_field(ggd.n_i_total, sampled["n"])
    _store_struct_field(ggd.t_i_average, sampled["ti"])
    _store_struct_field(ggd.psi, sampled["psi"])
    if "Zeff" in solution.parameters["physics"]:
        zeff_values = np.full(r_grid.shape, float(solution.parameters["physics"]["Zeff"]), dtype=float)
        zeff_values[np.isnan(sampled["n"])] = np.nan
        _store_struct_field(ggd.zeff, zeff_values)


def build_discharge_plasma_profiles_ids(timed_solutions, metadata: IMASExportMetadata, grid: RectangularGrid2D):
    import imas

    first_time, first_solution = timed_solutions[0]
    times = np.asarray([time_value for time_value, _ in timed_solutions], dtype=float)

    plasma = imas.IDSFactory().plasma_profiles()
    plasma.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    plasma.time = times
    populate_rectangular_grid_ggd_array(plasma.grid_ggd, grid, float(first_time))
    plasma.ggd.resize(len(timed_solutions))

    for index, (time_value, solution) in enumerate(timed_solutions):
        _populate_plasma_ggd(solution, plasma.ggd[index], time_value, grid)

    plasma.code.name = "SOLEDGE-HDG"
    plasma.code.repository = "hdg_postprocess"
    plasma.code.description = "Time-resolved plasma profiles exported from SOLEDGE-HDG by hdg_postprocess."
    metadata_dict = {
        "grid_shape": [grid.nr, grid.nz],
        "grid_r_range_m": [grid.r_min, grid.r_max],
        "grid_z_range_m": [grid.z_min, grid.z_max],
        "ggd_grid_name": "rectangular_rz",
        "ggd_grid_subset": "All exported plasma fields currently live on the nodes subset.",
        "value_ordering": "Node values are flattened from meshgrid(indexing='ij') in C order, so R is the slow axis and Z the fast axis.",
        "outside_mesh_policy": "Values outside the HDG mesh are exported as NaN.",
        "model_note": "SOLEDGE-HDG currently uses shared plasma density and parallel velocity for electrons and the single ion species.",
        "snapshot_count": len(timed_solutions),
        "exported_times_s": times.tolist(),
    }
    if "Zeff" in first_solution.parameters["physics"]:
        metadata_dict["Zeff"] = float(first_solution.parameters["physics"]["Zeff"])
    plasma.code.parameters = json.dumps(metadata_dict, sort_keys=True)
    return plasma


def write_discharge(entry, solutions, metadata: IMASExportMetadata, grid: RectangularGrid2D, *, include_summary=True, include_equilibrium=True, include_plasma_profiles=True, sort_by_time=True, time_getter=None):
    if not solutions:
        raise ValueError("Full-discharge export requires at least one solution snapshot.")
    if time_getter is None:
        raise ValueError("A time_getter callable must be provided for full-discharge export.")

    timed_solutions = _normalize_snapshot_times(solutions, sort_by_time=sort_by_time, time_getter=time_getter)
    times = [time_value for time_value, _ in timed_solutions]
    ordered_solutions = [solution for _, solution in timed_solutions]

    written = {}
    if include_summary:
        summary = _build_discharge_summary_ids(ordered_solutions, metadata, times)
        entry.put(summary, metadata.occurrence)
        written["summary"] = summary
    if include_equilibrium:
        equilibrium = build_discharge_equilibrium_ids(timed_solutions, metadata, grid)
        entry.put(equilibrium, metadata.occurrence)
        written["equilibrium"] = equilibrium
    if include_plasma_profiles:
        plasma = build_discharge_plasma_profiles_ids(timed_solutions, metadata, grid)
        entry.put(plasma, metadata.occurrence)
        written["plasma_profiles"] = plasma
    return written
