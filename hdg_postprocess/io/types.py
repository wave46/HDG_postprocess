from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class NormalizedSolutionData:
    raw_solutions: List[np.ndarray]
    raw_solutions_skeleton: List[np.ndarray]
    raw_gradients: List[np.ndarray]
    raw_equilibriums: List[Dict[str, np.ndarray]]
    raw_solution_boundary_infos: List[Dict[str, np.ndarray]]
    raw_transport_1d: Optional[List[Dict[str, np.ndarray]]] = None
    raw_neutral_flux_limiter_diagnostics: Optional[List[Dict[str, np.ndarray]]] = None
    raw_neutral_wall_source_diagnostics: Optional[List[Dict[str, np.ndarray]]] = None
    parameters: Dict = None
    n_partitions: int = 1
    mesh_path: str = ""
    mesh_name_base: str = ""


@dataclass
class NormalizedMeshData:
    raw_vertices: List[np.ndarray]
    raw_connectivity: List[np.ndarray]
    raw_connectivity_boundary: List[np.ndarray]
    raw_mesh_numbers: List[Dict[str, int]]
    raw_boundary_flags: List[np.ndarray]
    raw_ghost_elements: List[np.ndarray]
    raw_ghost_faces: List[np.ndarray]
    mesh_parameters: Dict
    n_partitions: int
    raw_rest_mesh_data: Optional[List[Dict[str, np.ndarray]]] = None
