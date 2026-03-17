from hdg_postprocess.api import load_reference_element
from hdg_postprocess.formats import load_from_file

from helpers import scenario_map


def test_mesh_ops_compatibility_surface(manifest_path, project_root):
    scenarios = scenario_map(manifest_path)
    cfg = scenarios["legacy_mesh_west"]
    mesh = load_from_file.load_HDG_mesh_from_file(cfg["mesh_path"], cfg["mesh_base"], cfg["n_partitions"])

    mesh.assembly.full()
    assert mesh.metadata.flags.combined_to_full

    _ = mesh.geometry.connectivity_big
    assert mesh.derived_geometry.connectivity_big is not None

    locator = mesh.geometry.element_locator
    probe = cfg["element_probe"]
    assert int(locator(probe[0], probe[1])) >= 0

    ref = load_reference_element(project_root / cfg["reference_element"])
    mesh.metadata.reference_element = ref
    _ = mesh.geometry.gauss_volumes
    assert mesh.derived_geometry.gauss_volumes is not None

    adjacent = mesh.geometry.adjacent_elements(int(locator(probe[0], probe[1])))
    assert len(adjacent) > 0
