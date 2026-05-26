import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hdg_postprocess.HDG_mesh import HDGmesh
from hdg_postprocess.HDG_solution import HDGsolution
from hdg_postprocess.core.solution.neutral_wall_sources import (
    check_neutral_wall_source_identities,
    collect_neutral_wall_source_totals,
    plot_element_node_field,
    read_neutral_wall_source_diagnostics,
)


def _parameters():
    return {
        "Neq": np.array([5]),
        "Ndim": np.array([2]),
        "switches": {"ohmicsrc": np.array([0])},
        "time": {"Current_time": np.array([0.0])},
        "adimensionalization": {
            "charge_scale": 1.0,
            "density_scale": 1.0,
            "length_scale": 1.0,
            "mass_scale": 1.0,
            "specific_energy_density_scale": 1.0,
            "temperature_scale": 1.0,
            "time_scale": 1.0,
        },
        "physics": {
            "Mref": 1.0,
            "conservative_variable_names": [b"rho", b"Gamma", b"nEi", b"nEe", b"rhon"],
            "physical_variable_names": [b"rho"],
        },
        "numerics": {},
    }


def _mesh():
    return HDGmesh(
        raw_vertices=[np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])],
        raw_connectivity=[np.array([[0, 1, 2], [2, 1, 3]])],
        raw_connectivity_boundary=[np.array([[0, 1], [2, 3]])],
        raw_mesh_numbers=[{"Nelems": 2, "Nextfaces": 2, "Nnodes": 4}],
        raw_boundary_flags=[],
        raw_ghost_elements=[],
        raw_ghost_faces=[],
        mesh_parameters={
            "Ndim": 2,
            "nodes_per_element": 3,
            "nodes_per_face": 2,
            "element_type": "triangle",
        },
        n_partitions=1,
    )


def _diagnostics():
    puff = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]])
    pump = np.array([[0.0, 0.5, 0.0], [0.25, 0.0, 1.0]])
    return {
        "element_puff_total": np.array([10.0]),
        "element_pump_total": np.array([2.0]),
        "element_net_total": np.array([8.0]),
        "puff_flux_density": puff.reshape(-1),
        "pump_flux_density": pump.reshape(-1),
        "net_flux_density": (puff - pump).reshape(-1),
    }


def _solution():
    u = np.zeros((2, 3, 5))
    q = np.zeros((2, 3, 5, 2))
    return HDGsolution(
        [u.reshape(-1)],
        [np.zeros((2 * 2 * 5,))],
        [q.reshape(-1)],
        [{"magnetic_field": np.ones((4, 3))}],
        [{}],
        _parameters(),
        1,
        _mesh(),
        raw_neutral_wall_source_diagnostics=[_diagnostics()],
    )


def test_neutral_wall_source_diagnostics_are_loaded_on_solution():
    sol = _solution()

    diagnostics = sol.neutral_wall_source_diagnostics
    assert diagnostics["element_puff_total"] == 10.0
    assert diagnostics["element_pump_total"] == 2.0
    assert diagnostics["element_net_total"] == 8.0
    assert diagnostics["net_flux_density"].shape == (2, 3)
    assert np.allclose(
        sol.neutrals.wall_source_diagnostic("net_flux_density"),
        diagnostics["puff_flux_density"] - diagnostics["pump_flux_density"],
    )
    assert sol.neutrals.wall_source_totals()["element_net_total"] == 8.0


def test_neutral_wall_source_identity_checks_pass():
    report = check_neutral_wall_source_identities(solution=_solution(), atol=1.0e-14, rtol=1.0e-14)

    assert report["net_flux_density"]["passed"]
    assert report["element_net_total"]["passed"]


def test_read_neutral_wall_source_diagnostics_from_hdf5(tmp_path):
    path = tmp_path / "wall_sources.h5"
    with h5py.File(path, "w") as h5:
        mesh = h5.create_group("mesh")
        mesh.create_dataset("Nelems", data=np.array([2]))
        mesh.create_dataset("Nnodesperelem", data=np.array([3]))
        group = h5.create_group("neutral_wall_sources_diagnostics")
        for key, value in _diagnostics().items():
            group.create_dataset(key, data=value)

    diagnostics = read_neutral_wall_source_diagnostics(path)

    assert diagnostics["puff_flux_density"].shape == (2, 3)
    assert diagnostics["element_net_total"] == 8.0


def test_collect_neutral_wall_source_totals(tmp_path):
    paths = []
    for idx in range(2):
        path = tmp_path / f"wall_sources_{idx}.h5"
        paths.append(path)
        diagnostics = _diagnostics()
        diagnostics["element_net_total"] = np.array([8.0 + idx])
        with h5py.File(path, "w") as h5:
            mesh = h5.create_group("mesh")
            mesh.create_dataset("Nelems", data=np.array([2]))
            mesh.create_dataset("Nnodesperelem", data=np.array([3]))
            group = h5.create_group("neutral_wall_sources_diagnostics")
            for key, value in diagnostics.items():
                group.create_dataset(key, data=value)

    rows = collect_neutral_wall_source_totals(paths)

    assert [row["element_net_total"] for row in rows] == [8.0, 9.0]


def test_plot_element_node_field_uses_discontinuous_element_layout():
    sol = _solution()
    fig, ax = plt.subplots()

    out_ax, artist = plot_element_node_field(
        sol,
        sol.neutrals.wall_source_diagnostic("net_flux_density"),
        ax=ax,
        label="net_flux_density",
    )

    assert out_ax is ax
    assert artist.get_array().shape[0] == 6
    plt.close(fig)
