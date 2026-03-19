#!/usr/bin/env python3

import argparse
from pathlib import Path

import imas


def build_summary(factory, description, comment, time):
    summary = factory.summary()
    summary.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    summary.ids_properties.comment = comment
    summary.description = description
    summary.time = [float(time)]
    return summary


def main():
    parser = argparse.ArgumentParser(description="Minimal IMAS DBEntry smoke test with the netCDF backend.")
    parser.add_argument("--output", default="tmp/imas_smoke.nc", help="Target IMAS netCDF path.")
    parser.add_argument("--time", type=float, default=0.0, help="Time value stored in the summary IDS.")
    parser.add_argument("--description", default="HDG IMAS smoke test", help="Summary description.")
    parser.add_argument("--comment", default="Minimal netCDF-backed IMAS write/read check.", help="Summary comment.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    factory = imas.IDSFactory()
    summary = build_summary(factory, args.description, args.comment, args.time)

    with imas.DBEntry(str(output_path), "x") as entry:
        entry.put(summary)

    with imas.DBEntry(str(output_path), "r") as entry:
        loaded = entry.get("summary")

    print("Wrote IMAS summary IDS to", output_path)
    print("description:", loaded.description)
    print("comment:", loaded.ids_properties.comment)
    print("time:", list(loaded.time))


if __name__ == "__main__":
    main()
