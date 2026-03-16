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


def test_mesh_ops_compatibility_surface(manifest_path, project_root):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["legacy_mesh_west"]
    mesh = load_from_file.load_HDG_mesh_from_file(cfg["mesh_path"], cfg["mesh_base"], cfg["n_partitions"])

    mesh.geometry.recombine_full()
    assert mesh.combined_to_full

    _ = mesh.geometry.connectivity_big
    assert mesh.connectivity_big is not None

    locator = mesh.geometry.element_locator
    probe = cfg["element_probe"]
    assert int(locator(probe[0], probe[1])) >= 0

    ref = _load_reference_element(project_root / cfg["reference_element"])
    mesh.reference_element = ref
    _ = mesh.geometry.gauss_volumes
    assert mesh.volumes_gauss is not None

    adjacent = mesh.geometry.adjacent_elements(int(locator(probe[0], probe[1])))
    assert len(adjacent) > 0
