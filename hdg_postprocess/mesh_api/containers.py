from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MeshFlags:
    combined_to_full: bool = False
    boundary_combined: bool = False
    connectivity_big_initialized: bool = False
    mask_initialized: bool = False
    element_locator_initialized: bool = False
    gauss_volumes_initialized: bool = False
    boundary_gauss_initialized: bool = False


@dataclass
class MeshCache:
    boundary_gauss_boundaries: Optional[Tuple[int, ...]] = None


@dataclass
class MeshMetadata:
    p_order: Optional[int] = None
    extent: dict = field(default_factory=dict)
    reference_element: Optional[dict] = None
    flags: MeshFlags = field(default_factory=MeshFlags)
    cache: MeshCache = field(default_factory=MeshCache)


@dataclass
class MeshGlobalState:
    vertices: object = None
    connectivity: object = None
    n_elements: Optional[int] = None
    n_vertices: Optional[int] = None
    n_faces: Optional[int] = None


@dataclass
class MeshDerivedGeometryState:
    connectivity_big: object = None
    mask: object = None
    element_locator: object = None
    vertices_gauss: object = None
    gauss_volumes: object = None


@dataclass
class MeshBoundaryState:
    connectivity: object = None
    flags: object = None
    face_element_number: object = None
    face_local_number: object = None
    filled: object = None
    indices: object = None
    vertices_gauss: object = None
    tangentials_gauss: object = None
    normals_gauss: object = None
    segment_length_gauss: object = None
    segment_surface_gauss: object = None
