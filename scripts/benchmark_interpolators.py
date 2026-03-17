#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.io

from hdg_postprocess.formats import load_from_file


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tests" / "baseline_manifest.json"


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text())


def load_reference_element(path):
    ref_elem = scipy.io.loadmat(path)
    if "refEl" in ref_elem:
        name = "refEl"
    elif "referenceelement" in ref_elem:
        name = "referenceelement"
    else:
        raise KeyError(f"Unsupported reference-element keys in {path}: {sorted(ref_elem.keys())}")

    ref_dic = {}
    ref_dic["IPcoordinates"] = ref_elem[name][0, 0][0]
    ref_dic["IPweights"] = ref_elem[name][0, 0][1][:, 0]
    ref_dic["N"] = ref_elem[name][0, 0][2]
    ref_dic["Nxi"] = ref_elem[name][0, 0][3]
    ref_dic["Neta"] = ref_elem[name][0, 0][4]
    ref_dic["IPcoordinates1d"] = ref_elem[name][0, 0][5]
    ref_dic["IPweights1d"] = ref_elem[name][0, 0][6]
    ref_dic["N1d"] = ref_elem[name][0, 0][7]
    ref_dic["N1dxi"] = ref_elem[name][0, 0][8]
    ref_dic["faceNodes"] = ref_elem[name][0, 0][9] - 1
    ref_dic["innerNodes"] = ref_elem[name][0, 0][10]
    ref_dic["faceNodes1d"] = ref_elem[name][0, 0][11] - 1
    ref_dic["NodesCoord"] = ref_elem[name][0, 0][12]
    ref_dic["NodesCoord1d"] = ref_elem[name][0, 0][13]
    ref_dic["degree"] = ref_elem[name][0, 0][14]
    return ref_dic


def load_solution_for_scenario(config):
    if config["kind"] != "solution":
        raise ValueError(f"Interpolator benchmark requires a solution scenario, got {config['kind']!r}")
    solution = load_from_file.load_HDG_solution_from_file(
        config["solution_path"],
        config["solution_base"],
        config.get("mesh_path"),
        config.get("mesh_base"),
        config["n_partitions"],
    )
    solution.mesh.metadata.reference_element = load_reference_element(ROOT / config["reference_element"])
    return solution


def build_query_sets(solution, unique_points, repeated_points):
    solution.mesh.geometry.connectivity_big
    triangles = solution.mesh.global_state.vertices[solution.mesh.derived_geometry.connectivity_big]
    centroids = triangles.mean(axis=1)
    stride = max(1, len(centroids) // unique_points)
    unique = [tuple(map(float, point)) for point in centroids[::stride][:unique_points]]
    if len(unique) < unique_points:
        raise ValueError(f"Requested {unique_points} unique points, got only {len(unique)}")
    repeated = unique[: min(repeated_points, len(unique))]
    return unique, repeated


def benchmark_callable(fn, points, repeat):
    start = time.perf_counter()
    for _ in range(repeat):
        for point in points:
            fn(*point)
    elapsed = time.perf_counter() - start
    return elapsed, len(points) * repeat


def benchmark_single_pass(fn, points):
    start = time.perf_counter()
    for point in points:
        fn(*point)
    elapsed = time.perf_counter() - start
    return elapsed, len(points)


def clear_interpolator_cache(interpolator):
    interpolator._hashed_shape_functions.clear()
    interpolator._hashed_shape_functions_dx.clear()
    interpolator._hashed_shape_functions_dy.clear()
    interpolator._hashed_element.clear()


def mixed_value_gradient(interpolator):
    def runner(x, y):
        interpolator.evaluate(x, y)
        interpolator.gradient(x, y)

    return runner


def format_result(name, elapsed, queries):
    qps = queries / elapsed if elapsed > 0 else float("inf")
    print(f"{name}_seconds={elapsed:.6f}")
    print(f"{name}_queries={queries}")
    print(f"{name}_queries_per_second={qps:.2f}")


def run_cold_warm_pair(name, fn, points):
    cold_elapsed, cold_queries = benchmark_single_pass(fn, points)
    format_result(f"{name}_cold", cold_elapsed, cold_queries)

    warm_elapsed, warm_queries = benchmark_single_pass(fn, points)
    format_result(f"{name}_warm", warm_elapsed, warm_queries)


def main():
    parser = argparse.ArgumentParser(description="Benchmark current interpolator value and gradient workloads.")
    parser.add_argument("--scenario", default="legacy_first", help="Solution scenario id from tests/baseline_manifest.json")
    parser.add_argument("--repeat", type=int, default=4, help="How many times to traverse each point set.")
    parser.add_argument("--unique-points", type=int, default=2000, help="Number of unique points to sample.")
    parser.add_argument("--repeated-points", type=int, default=64, help="Number of points in the repeated cache-friendly set.")
    args = parser.parse_args()

    manifest = load_manifest()
    scenarios = {scenario["id"]: scenario for scenario in manifest["scenarios"]}
    config = scenarios[args.scenario]

    solution = load_solution_for_scenario(config)
    solution.sample.define_interpolators()

    unique_points, repeated_points = build_query_sets(solution, args.unique_points, args.repeated_points)
    value_interpolator = solution.interpolators.solution[0]

    print(f"scenario={args.scenario}")
    print(f"unique_points={len(unique_points)}")
    print(f"repeated_points={len(repeated_points)}")

    mixed = mixed_value_gradient(value_interpolator)

    for workload_name, fn in (
        ("value_unique", value_interpolator.evaluate),
        ("value_repeated", value_interpolator.evaluate),
        ("gradient_unique", value_interpolator.gradient),
        ("gradient_repeated", value_interpolator.gradient),
        ("mixed_unique", mixed),
        ("mixed_repeated", mixed),
    ):
        points = unique_points if "unique" in workload_name else repeated_points
        clear_interpolator_cache(value_interpolator)
        run_cold_warm_pair(workload_name, fn, points)

    for workload_name, fn in (
        ("value_unique", value_interpolator.evaluate),
        ("value_repeated", value_interpolator.evaluate),
        ("gradient_unique", value_interpolator.gradient),
        ("gradient_repeated", value_interpolator.gradient),
        ("mixed_unique", mixed),
        ("mixed_repeated", mixed),
    ):
        points = unique_points if "unique" in workload_name else repeated_points
        clear_interpolator_cache(value_interpolator)
        elapsed, queries = benchmark_callable(fn, points, args.repeat)
        format_result(f"{workload_name}_mixed", elapsed, queries)


if __name__ == "__main__":
    main()
