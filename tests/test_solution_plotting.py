import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from hdg_postprocess.formats import load_from_file

from helpers import scenario_map


def _load_reference_element(path):
    ref_elem = scipy.io.loadmat(path)
    key = "refEl" if "refEl" in ref_elem else "referenceelement"
    ref_dic = {}
    ref_dic["IPcoordinates"] = ref_elem[key][0, 0][0]
    ref_dic["IPweights"] = ref_elem[key][0, 0][1][:, 0]
    ref_dic["N"] = ref_elem[key][0, 0][2]
    ref_dic["Nxi"] = ref_elem[key][0, 0][3]
    ref_dic["Neta"] = ref_elem[key][0, 0][4]
    ref_dic["IPcoordinates1d"] = ref_elem[key][0, 0][5]
    ref_dic["IPweights1d"] = ref_elem[key][0, 0][6]
    ref_dic["N1d"] = ref_elem[key][0, 0][7]
    ref_dic["N1dxi"] = ref_elem[key][0, 0][8]
    ref_dic["faceNodes"] = ref_elem[key][0, 0][9] - 1
    ref_dic["innerNodes"] = ref_elem[key][0, 0][10]
    ref_dic["faceNodes1d"] = ref_elem[key][0, 0][11] - 1
    ref_dic["NodesCoord"] = ref_elem[key][0, 0][12]
    ref_dic["NodesCoord1d"] = ref_elem[key][0, 0][13]
    ref_dic["degree"] = ref_elem[key][0, 0][14]
    return ref_dic


def test_solution_plotting_smoke_and_mesh_linewidth(manifest_path):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["embedded_k_model"]

    sol = load_from_file.load_HDG_solution_from_file(
        cfg["solution_path"],
        cfg["solution_base"],
        cfg.get("mesh_path"),
        cfg.get("mesh_base"),
        cfg["n_partitions"],
    )
    sol.mesh.reference_element = _load_reference_element(cfg["reference_element"])

    fig1, axes1, conservative = sol.plot_overview(n_levels=5)
    fig2, axes2, physical = sol.plot_overview_physical(n_levels=5)

    assert conservative.shape[1] == sol.neq
    assert physical.shape[0] == sol.views.simple.solution.conservative.shape[0]
    assert np.asarray(axes1).size >= sol.neq
    assert np.asarray(axes2).size >= sol.neq

    fig3, ax3 = plt.subplots()
    sol.assembly.full()
    ax3 = sol.mesh.plot_full_mesh(ax=ax3, linewidth=2.75)
    assert ax3.collections, "expected a mesh collection to be added"
    collection = ax3.collections[0]
    assert np.allclose(collection.get_linewidths(), 2.75)

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
