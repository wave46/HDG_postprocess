#!/usr/bin/env python3

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from hdg_postprocess.imas_export.reader import (
    equilibrium_slice,
    field_style_presets,
    format_time_labels,
    load_ids,
    make_field_animation,
    plasma_slice,
    plot_field_2d,
    save_animation,
    symmetric_limits,
)


FIELD_LABELS = {
    "ne": "n_e [m^-3]",
    "nn": "n_n [m^-3]",
    "te": "T_e [eV]",
    "ti": "T_i [eV]",
    "u_par": "u_parallel [m/s]",
    "psi": "psi",
    "br": "B_R [T]",
    "bz": "B_Z [T]",
    "bphi": "B_phi [T]",
    "jphi": "j_phi",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render animations or per-frame PNGs from SOLEDGE-HDG IMAS exports."
    )
    parser.add_argument("db_path", help="Path to the IMAS netCDF file.")
    parser.add_argument(
        "--field",
        nargs="+",
        default=None,
        choices=sorted(FIELD_LABELS),
        help="Saved field(s) to visualize. If omitted, all supported fields are rendered.",
    )
    parser.add_argument("--occurrence", type=int, default=0, help="IDS occurrence to read.")
    parser.add_argument("--start", type=int, default=0, help="First time index to include.")
    parser.add_argument("--stop", type=int, default=None, help="Stop time index (exclusive).")
    parser.add_argument("--step", type=int, default=1, help="Stride between time indices.")
    parser.add_argument(
        "--output",
        default="imas_field_animation.mp4",
        help="Output animation path. Used when --frames-dir is not set.",
    )
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Optional directory for per-frame PNG export. If set, frames are rendered instead of a single video.",
    )
    parser.add_argument("--fps", type=int, default=8, help="Animation frame rate.")
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI for saved output.")
    parser.add_argument(
        "--separatrix-level",
        type=float,
        default=None,
        help="Optional psi contour level to overlay as a separatrix guide.",
    )
    parser.add_argument(
        "--cmap",
        default=None,
        help="Override the default colormap for the selected field.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Worker count for per-frame PNG rendering. "
            "Animation saving remains serial; this only applies with --frames-dir."
        ),
    )
    return parser.parse_args()


def positive_log_norm(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        return None
    return LogNorm(vmin=float(finite.min()), vmax=float(finite.max()))


def build_norm(field_name, frames):
    style = field_style_presets()[field_name]
    values = np.asarray(frames, dtype=float)
    if style["scale"] == "log":
        return positive_log_norm(values)

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    if style.get("symmetric"):
        vmin, vmax = symmetric_limits(finite)
        return Normalize(vmin=vmin, vmax=vmax)
    return Normalize(vmin=float(finite.min()), vmax=float(finite.max()))


def frame_norm_payload(field_name, frames):
    norm = build_norm(field_name, frames)
    if norm is None:
        return {"kind": "none"}
    if isinstance(norm, LogNorm):
        return {"kind": "log", "vmin": float(norm.vmin), "vmax": float(norm.vmax)}
    return {"kind": "linear", "vmin": float(norm.vmin), "vmax": float(norm.vmax)}


def norm_from_payload(payload):
    kind = payload["kind"]
    if kind == "none":
        return None
    if kind == "log":
        return LogNorm(vmin=payload["vmin"], vmax=payload["vmax"])
    return Normalize(vmin=payload["vmin"], vmax=payload["vmax"])


def load_selected_frames(db_path, *, occurrence, field_name, time_indices, separatrix_level):
    _, equilibrium, plasma = load_ids(db_path, occurrence=occurrence)

    frames = []
    times = []
    r_frames = []
    z_frames = []
    separatrix_frames = []

    for time_index in time_indices:
        eq = equilibrium_slice(equilibrium, plasma, time_index=time_index)
        pl = plasma_slice(plasma, time_index=time_index)
        data = eq if field_name in {"psi", "br", "bz", "bphi", "jphi"} else pl

        field = data.get(field_name)
        if field is None:
            raise ValueError(f"Field '{field_name}' is not available in the selected export.")

        frames.append(field)
        times.append(float(eq["time"]))
        r_frames.append(eq["r"])
        z_frames.append(eq["z"])
        if separatrix_level is not None:
            separatrix_frames.append({"psi": eq["psi"], "level": float(separatrix_level)})

    return {
        "frames": frames,
        "times": times,
        "r_frames": r_frames,
        "z_frames": z_frames,
        "separatrix_frames": separatrix_frames if separatrix_frames else None,
    }


def render_png_frame(payload):
    matplotlib.use("Agg")
    import matplotlib.pyplot as local_plt

    norm = norm_from_payload(payload["norm"])
    fig, ax = local_plt.subplots(figsize=(6, 8))
    mesh, label = plot_field_2d(
        ax,
        r=payload["r"],
        z=payload["z"],
        field=payload["field"],
        title=payload["title"],
        label=payload["label"],
        cmap=payload["cmap"],
        norm=norm,
        separatrix=payload["separatrix"],
    )
    fig.colorbar(mesh, ax=ax, label=label)
    fig.savefig(payload["path"], dpi=payload["dpi"], bbox_inches="tight")
    local_plt.close(fig)
    return payload["path"]


def export_png_frames(
    frames_dir,
    *,
    field_name,
    frames,
    times,
    r_frames,
    z_frames,
    separatrix_frames,
    cmap,
    dpi,
    workers,
):
    frames_dir.mkdir(parents=True, exist_ok=True)
    norm_payload = frame_norm_payload(field_name, frames)
    time_labels = format_time_labels(times)

    payloads = []
    for index, (field, time_label, r_grid, z_grid) in enumerate(zip(frames, time_labels, r_frames, z_frames)):
        payloads.append(
            {
                "r": r_grid,
                "z": z_grid,
                "field": field,
                "title": f"{field_name} @ t={time_label} s",
                "label": FIELD_LABELS[field_name],
                "cmap": cmap,
                "norm": norm_payload,
                "separatrix": None if separatrix_frames is None else separatrix_frames[index],
                "path": str(frames_dir / f"{field_name}_{index:05d}.png"),
                "dpi": int(dpi),
            }
        )

    if int(workers) <= 1:
        for payload in payloads:
            render_png_frame(payload)
        return

    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        list(executor.map(render_png_frame, payloads))


def resolve_output_path(output, field_name, *, multiple_fields):
    path = Path(output)
    if not multiple_fields:
        return path
    return path.with_name(f"{path.stem}_{field_name}{path.suffix}")


def main():
    args = parse_args()
    styles = field_style_presets()
    selected_fields = list(args.field) if args.field is not None else list(FIELD_LABELS)

    _, equilibrium, _ = load_ids(args.db_path, occurrence=args.occurrence)
    total_times = len(equilibrium.time)
    stop = total_times if args.stop is None else min(int(args.stop), total_times)
    time_indices = list(range(int(args.start), stop, int(args.step)))
    if not time_indices:
        raise ValueError("No time indices selected for rendering.")

    for field_name in selected_fields:
        cmap = args.cmap or styles[field_name]["cmap"]
        series = load_selected_frames(
            args.db_path,
            occurrence=args.occurrence,
            field_name=field_name,
            time_indices=time_indices,
            separatrix_level=args.separatrix_level,
        )

        if args.frames_dir:
            frames_dir = Path(args.frames_dir)
            if len(selected_fields) > 1:
                frames_dir = frames_dir / field_name
            export_png_frames(
                frames_dir,
                field_name=field_name,
                frames=series["frames"],
                times=series["times"],
                r_frames=series["r_frames"],
                z_frames=series["z_frames"],
                separatrix_frames=series["separatrix_frames"],
                cmap=cmap,
                dpi=args.dpi,
                workers=args.workers,
            )
            print(f"Saved {len(series['frames'])} PNG frames to {frames_dir}")
            continue

        norm = build_norm(field_name, series["frames"])
        fig, animation = make_field_animation(
            frames=series["frames"],
            times=series["times"],
            title=field_name,
            label=FIELD_LABELS[field_name],
            cmap=cmap,
            norm=norm,
            separatrix_frames=series["separatrix_frames"],
            r_frames=series["r_frames"],
            z_frames=series["z_frames"],
        )
        output_path = resolve_output_path(args.output, field_name, multiple_fields=len(selected_fields) > 1)
        try:
            save_animation(animation, output_path, fps=args.fps, dpi=args.dpi)
        finally:
            plt.close(fig)
        print(f"Saved animation to {output_path}")


if __name__ == "__main__":
    main()
