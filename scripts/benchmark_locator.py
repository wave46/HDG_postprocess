#!/usr/bin/env python3

import argparse
import json
import importlib
import time
from pathlib import Path

import numpy as np
from raysect.core.math.function.float import Discrete2DMesh

from hdg_postprocess.formats import load_from_file


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tests" / "baseline_manifest.json"


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text())


def load_mesh_for_scenario(config):
    if config["kind"] == "mesh":
        return load_from_file.load_HDG_mesh_from_file(
            config["mesh_path"],
            config["mesh_base"],
            config["n_partitions"],
        )
    solution = load_from_file.load_HDG_solution_from_file(
        config["solution_path"],
        config["solution_base"],
        config.get("mesh_path"),
        config.get("mesh_base"),
        config["n_partitions"],
    )
    return solution.mesh


def build_query_points(mesh, n_r=600, n_z=32):
    triangles = mesh.vertices_glob[mesh.connectivity_big]
    centroids = triangles.mean(axis=1)
    sample_step = max(1, len(centroids) // n_r)
    return [tuple(map(float, point)) for point in centroids[::sample_step]]


def benchmark_callable(name, fn, points, repeat):
    start = time.perf_counter()
    for _ in range(repeat):
        for point in points:
            fn(*point)
    elapsed = time.perf_counter() - start
    return {"name": name, "seconds": elapsed, "queries": len(points) * repeat}


def maybe_load_native_locator():
    try:
        module = importlib.import_module("hdg_postprocess.locator")
    except ModuleNotFoundError:
        return None
    return getattr(module, "Exact2DMeshFunction", None)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Exact2DMeshFunction against raysect Discrete2DMesh.")
    parser.add_argument("--scenario", default="legacy_mesh_west", help="Scenario id from tests/baseline_manifest.json")
    parser.add_argument("--repeat", type=int, default=3, help="How many times to traverse the query set.")
    args = parser.parse_args()

    manifest = load_manifest()
    scenarios = {scenario["id"]: scenario for scenario in manifest["scenarios"]}
    config = scenarios[args.scenario]

    mesh = load_mesh_for_scenario(config)
    mesh.create_connectivity_big()
    repeats = mesh.connectivity_big.shape[0] // mesh.connectivity_glob.shape[0]
    element_numbers = np.repeat(np.arange(len(mesh.connectivity_glob)), repeats)

    raysect_locator = Discrete2DMesh(
        mesh.vertices_glob,
        mesh.connectivity_big,
        element_numbers,
        limit=False,
        default_value=-1,
    )

    points = build_query_points(mesh)

    raysect_result = benchmark_callable("raysect", raysect_locator, points, args.repeat)

    print(f"scenario={args.scenario}")
    print(f"queries={raysect_result['queries']}")
    print(f"raysect_seconds={raysect_result['seconds']:.6f}")

    native_locator_cls = maybe_load_native_locator()
    if native_locator_cls is None:
        print("native_locator=unavailable")
        return

    native_locator = native_locator_cls(
        mesh.vertices_glob,
        mesh.connectivity_big,
        element_numbers,
        limit=False,
        default_value=-1,
    )
    native_result = benchmark_callable("native", native_locator, points, args.repeat)
    print(f"native_seconds={native_result['seconds']:.6f}")
    if raysect_result["seconds"] > 0:
        print(f"speed_ratio_native_over_raysect={native_result['seconds'] / raysect_result['seconds']:.6f}")


if __name__ == "__main__":
    main()
