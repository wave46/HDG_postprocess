#!/usr/bin/env python3

import argparse
import cProfile
import io
import json
import pstats
from pathlib import Path

from benchmark_interpolators import (
    MANIFEST_PATH,
    build_query_sets,
    clear_interpolator_cache,
    load_solution_for_scenario,
)


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text())


def build_workloads(interpolator):
    def mixed(x, y):
        interpolator.evaluate(x, y)
        interpolator.gradient(x, y)

    return {
        "value": interpolator.evaluate,
        "gradient": interpolator.gradient,
        "mixed": mixed,
    }


def run_cold_pass(fn, points):
    for x, y in points:
        fn(x, y)


def main():
    parser = argparse.ArgumentParser(description="Profile a single cold interpolator pass.")
    parser.add_argument("--scenario", default="legacy_first", help="Solution scenario id from tests/baseline_manifest.json")
    parser.add_argument("--workload", choices=("value", "gradient", "mixed"), default="value")
    parser.add_argument("--unique-points", type=int, default=200, help="Number of unique interior points to profile.")
    parser.add_argument("--repeated-points", type=int, default=32, help="Number of points in the repeated set.")
    parser.add_argument("--sort", default="cumtime", help="pstats sort key")
    parser.add_argument("--limit", type=int, default=20, help="How many profile rows to print")
    args = parser.parse_args()

    manifest = load_manifest()
    scenarios = {scenario["id"]: scenario for scenario in manifest["scenarios"]}
    solution = load_solution_for_scenario(scenarios[args.scenario])
    solution.sample.define_interpolators()

    unique_points, _ = build_query_sets(solution, args.unique_points, args.repeated_points)
    interpolator = solution.interpolators.solution[0]
    workloads = build_workloads(interpolator)
    clear_interpolator_cache(interpolator)

    profiler = cProfile.Profile()
    profiler.enable()
    run_cold_pass(workloads[args.workload], unique_points)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(args.sort)
    stats.print_stats(args.limit)

    print(f"scenario={args.scenario}")
    print(f"workload={args.workload}")
    print(f"unique_points={len(unique_points)}")
    print(stream.getvalue())


if __name__ == "__main__":
    main()
