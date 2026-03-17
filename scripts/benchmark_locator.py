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
    triangles = mesh.global_state.vertices[mesh.derived_geometry.connectivity_big]
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


def benchmark_native_locator(native_locator_cls, mesh, element_numbers, points, repeat, leaf_size, diagnostics=False):
    build_start = time.perf_counter()
    native_locator = native_locator_cls(
        mesh.global_state.vertices,
        mesh.derived_geometry.connectivity_big,
        element_numbers,
        limit=False,
        default_value=-1,
        leaf_size=leaf_size,
        collect_stats=diagnostics,
    )
    native_build_seconds = time.perf_counter() - build_start
    native_result = benchmark_callable("native", native_locator, points, repeat)
    result = {
        "leaf_size": leaf_size,
        "build_seconds": native_build_seconds,
        "query_seconds": native_result["seconds"],
        "total_seconds": native_build_seconds + native_result["seconds"],
    }
    if diagnostics:
        result["statistics"] = native_locator.statistics()
    return result


def maybe_load_native_locator():
    try:
        module = importlib.import_module("hdg_postprocess.locator")
    except ModuleNotFoundError:
        return None
    return getattr(module, "Exact2DMeshFunction", None)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Exact2DMeshFunction against raysect Discrete2DMesh.")
    parser.add_argument("--scenario", default="legacy_mesh_west", help="Scenario id from tests/baseline_manifest.json")
    parser.add_argument("--repeat", type=int, default=30, help="How many times to traverse the query set.")
    parser.add_argument("--leaf-size", type=int, default=8, help="Leaf size for the native tree locator.")
    parser.add_argument(
        "--leaf-sweep",
        default="",
        help="Comma-separated native leaf sizes to benchmark, e.g. 4,8,16,32.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print native locator traversal diagnostics.",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    scenarios = {scenario["id"]: scenario for scenario in manifest["scenarios"]}
    config = scenarios[args.scenario]

    mesh = load_mesh_for_scenario(config)
    mesh.geometry.connectivity_big
    repeats = mesh.derived_geometry.connectivity_big.shape[0] // mesh.global_state.connectivity.shape[0]
    element_numbers = np.repeat(np.arange(len(mesh.global_state.connectivity)), repeats)

    build_start = time.perf_counter()
    raysect_locator = Discrete2DMesh(
        mesh.global_state.vertices,
        mesh.derived_geometry.connectivity_big,
        element_numbers,
        limit=False,
        default_value=-1,
    )
    raysect_build_seconds = time.perf_counter() - build_start

    points = build_query_points(mesh)

    raysect_result = benchmark_callable("raysect", raysect_locator, points, args.repeat)

    print(f"scenario={args.scenario}")
    print(f"queries={raysect_result['queries']}")
    print(f"raysect_build_seconds={raysect_build_seconds:.6f}")
    print(f"raysect_seconds={raysect_result['seconds']:.6f}")
    print(f"raysect_total_seconds={raysect_build_seconds + raysect_result['seconds']:.6f}")

    native_locator_cls = maybe_load_native_locator()
    if native_locator_cls is None:
        print("native_locator=unavailable")
        return

    leaf_sizes = [args.leaf_size]
    if args.leaf_sweep:
        leaf_sizes = [int(value) for value in args.leaf_sweep.split(",") if value.strip()]

    native_results = [
        benchmark_native_locator(
            native_locator_cls,
            mesh,
            element_numbers,
            points,
            args.repeat,
            leaf_size,
            diagnostics=args.diagnostics,
        )
        for leaf_size in leaf_sizes
    ]

    for native_result in native_results:
        prefix = "native" if len(native_results) == 1 else f"native_leaf_{native_result['leaf_size']}"
        print(f"{prefix}_build_seconds={native_result['build_seconds']:.6f}")
        print(f"{prefix}_seconds={native_result['query_seconds']:.6f}")
        print(f"{prefix}_total_seconds={native_result['total_seconds']:.6f}")
        if raysect_result["seconds"] > 0:
            print(
                f"{prefix}_speed_ratio_native_over_raysect="
                f"{native_result['query_seconds'] / raysect_result['seconds']:.6f}"
            )
        if raysect_build_seconds > 0:
            print(
                f"{prefix}_build_ratio_native_over_raysect="
                f"{native_result['build_seconds'] / raysect_build_seconds:.6f}"
            )
        if args.diagnostics and "statistics" in native_result:
            for key, value in native_result["statistics"].items():
                print(f"{prefix}_{key}={value}")

    if len(native_results) > 1:
        best_total = min(native_results, key=lambda result: result["total_seconds"])
        best_query = min(native_results, key=lambda result: result["query_seconds"])
        print(f"best_native_leaf_by_total={best_total['leaf_size']}")
        print(f"best_native_leaf_by_query={best_query['leaf_size']}")


if __name__ == "__main__":
    main()
