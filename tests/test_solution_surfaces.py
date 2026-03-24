import numpy as np

from hdg_postprocess.formats import load_from_file
from hdg_postprocess.api import configure_solution_setup

from helpers import require_scenario_data, scenario_map


def test_surface_te_methods_return_finite_values(manifest_path):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["power_balance_with_cooling"]
    require_scenario_data(cfg)

    sol = load_from_file.load_HDG_solution_from_file(
        cfg["solution_path"],
        cfg["solution_base"],
        cfg.get("mesh_path"),
        cfg.get("mesh_base"),
        cfg["n_partitions"],
    )

    configure_solution_setup(
        sol,
        reference_element=cfg["reference_element"],
        radiation_model=cfg["radiation_model"],
        atomic_data_dir="demos/data/atomic",
        neutral_diffusion=True,
    )
    sol.parameters["physics"]["R_E"] = cfg["r_e_override"]

    te_node = sol.flux_surface.te(0.8, method="node_band", width=2e-3)
    te_gauss = sol.flux_surface.te(0.8, method="gauss_band", width=2e-3)
    te_shell = sol.flux_surface.te(0.8, method="gauss_shell", width=2e-3)

    assert np.isfinite(te_node)
    assert np.isfinite(te_gauss)
    assert np.isfinite(te_shell)
