import json
from pathlib import Path
import shutil

import numpy as np


def load_ids(db_path, occurrence=0):
    import imas

    with imas.DBEntry(str(db_path), "r") as entry:
        summary = entry.get("summary", occurrence)
        equilibrium = entry.get("equilibrium", occurrence)
        plasma = entry.get("plasma_profiles", occurrence)
    return summary, equilibrium, plasma


def resolve_plasma_grid_entry(plasma, time_index):
    grid_entry = plasma.grid_ggd[int(time_index)]
    seen = set()
    while str(grid_entry.path):
        path = str(grid_entry.path)
        if path in seen:
            raise RuntimeError(f"Circular IMAS grid reference: {path}")
        seen.add(path)
        ref_index = int(path.rsplit("grid_ggd(", 1)[1].rstrip(")")) - 1
        grid_entry = plasma.grid_ggd[ref_index]
    return grid_entry


def grid_coordinates(grid_ggd_entry, grid_shape):
    nr, nz = grid_shape
    nodes = grid_ggd_entry.space[0].objects_per_dimension[0].object
    coords = np.array([node.geometry for node in nodes], dtype=float)
    return coords[:, 0].reshape(nr, nz), coords[:, 1].reshape(nr, nz)


def equilibrium_slice(equilibrium, plasma, time_index=0):
    params = json.loads(str(equilibrium.code.parameters))
    nr, nz = params["grid_shape"]
    eq_ggd = equilibrium.time_slice[int(time_index)].ggd[0]
    r_grid, z_grid = grid_coordinates(resolve_plasma_grid_entry(plasma, time_index), (nr, nz))
    return {
        "params": params,
        "time": float(equilibrium.time[int(time_index)]),
        "r": r_grid,
        "z": z_grid,
        "psi": eq_ggd.psi[0].values.reshape(nr, nz),
        "br": eq_ggd.b_field_r[0].values.reshape(nr, nz),
        "bz": eq_ggd.b_field_z[0].values.reshape(nr, nz),
        "bphi": eq_ggd.b_field_phi[0].values.reshape(nr, nz),
        "jphi": None if len(eq_ggd.j_phi) == 0 else eq_ggd.j_phi[0].values.reshape(nr, nz),
    }


def plasma_slice(plasma, time_index=0):
    params = json.loads(str(plasma.code.parameters))
    nr, nz = params["grid_shape"]
    plasma_ggd = plasma.ggd[int(time_index)]
    ion = plasma_ggd.ion[0]
    neutral = plasma_ggd.neutral[0]
    return {
        "params": params,
        "time": float(plasma.time[int(time_index)]),
        "ne": plasma_ggd.electrons.density[0].values.reshape(nr, nz),
        "te": plasma_ggd.electrons.temperature[0].values.reshape(nr, nz),
        "ti": ion.temperature[0].values.reshape(nr, nz),
        "u_par": ion.velocity[0].parallel.reshape(nr, nz),
        "nn": neutral.density[0].values.reshape(nr, nz),
        "psi": plasma_ggd.psi[0].values.reshape(nr, nz),
        "zeff": np.asarray(plasma.global_quantities.z_eff_resistive),
    }


def symmetric_limits(field):
    finite = np.asarray(field, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    max_abs = float(np.max(np.abs(finite)))
    return -max_abs, max_abs


def format_time_labels(times, *, precision=3):
    values = np.asarray(times, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [str(time) for time in times]
    max_abs = float(np.max(np.abs(finite)))
    if max_abs >= 1.0e4 or (max_abs > 0.0 and max_abs < 1.0e-2):
        width = precision + 7
        return [f"{float(time):{width}.{precision}e}" for time in values]

    max_int_digits = max(1, int(np.floor(np.log10(max_abs))) + 1) if max_abs > 0.0 else 1
    width = max_int_digits + precision + 3
    return [f"{float(time):{width}.{precision}f}" for time in values]


def field_style_presets():
    return {
        "ne": {"cmap": "magma", "scale": "log"},
        "nn": {"cmap": "magma", "scale": "log"},
        "te": {"cmap": "magma", "scale": "log"},
        "ti": {"cmap": "magma", "scale": "log"},
        "psi": {"cmap": "cividis", "scale": "linear"},
        "br": {"cmap": "RdBu_r", "scale": "linear", "symmetric": True},
        "bz": {"cmap": "RdBu_r", "scale": "linear", "symmetric": True},
        "bphi": {"cmap": "viridis", "scale": "linear"},
        "jphi": {"cmap": "RdBu_r", "scale": "linear", "symmetric": True},
        "u_par": {"cmap": "RdBu_r", "scale": "linear", "symmetric": True},
        "M": {"cmap": "RdBu_r", "scale": "linear", "symmetric": True},
    }


def plot_field_2d(
    ax,
    *,
    r,
    z,
    field,
    title,
    label,
    cmap,
    norm=None,
    separatrix=None,
    separatrix_kwargs=None,
):
    mesh = ax.pcolormesh(r, z, field, shading="auto", cmap=cmap, norm=norm)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    ax.set_title(title)
    if separatrix is not None:
        kwargs = {"colors": "white", "linewidths": 1.0}
        if separatrix_kwargs:
            kwargs.update(separatrix_kwargs)
        ax.contour(r, z, separatrix["psi"], levels=[separatrix["level"]], **kwargs)
    return mesh, label


def make_field_animation(
    *,
    frames,
    times,
    title,
    label,
    cmap,
    norm=None,
    separatrix_frames=None,
    r=None,
    z=None,
    r_frames=None,
    z_frames=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.animation import FuncAnimation

    if r_frames is None or z_frames is None:
        if r is None or z is None:
            raise ValueError("Provide either fixed r/z or per-frame r_frames/z_frames for animation.")
    elif len(r_frames) != len(frames) or len(z_frames) != len(frames):
        raise ValueError("r_frames and z_frames must have the same length as frames.")
    time_labels = format_time_labels(times)

    fig, ax = plt.subplots(figsize=(6, 8))
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=label)

    def update(frame_index):
        ax.clear()
        current_r = r if r_frames is None else r_frames[frame_index]
        current_z = z if z_frames is None else z_frames[frame_index]
        mesh = ax.pcolormesh(current_r, current_z, frames[frame_index], shading="auto", cmap=cmap, norm=norm)
        ax.set_title(f"{title} @ t={time_labels[frame_index]} s")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")
        if separatrix_frames is not None:
            sep = separatrix_frames[frame_index]
            ax.contour(
                current_r,
                current_z,
                sep["psi"],
                levels=[sep["level"]],
                colors="white",
                linewidths=1.0,
            )
        return (mesh,)

    animation = FuncAnimation(fig, update, frames=len(frames), interval=150, blit=False)
    update(0)
    return fig, animation


def save_animation(animation, path, *, fps=8, dpi=140, writer=None):
    path = Path(path)
    kwargs = {"fps": int(fps), "dpi": int(dpi)}
    if writer is None:
        suffix = path.suffix.lower()
        if suffix == ".mp4":
            if shutil.which("ffmpeg") is None:
                raise RuntimeError(
                    "Saving MP4 animations requires 'ffmpeg' to be available in PATH. "
                    "Install ffmpeg on the cluster, choose a .gif output, or use --frames-dir."
                )
            writer = "ffmpeg"
        elif suffix == ".gif":
            writer = "pillow"
    if writer is not None:
        kwargs["writer"] = writer
    animation.save(str(path), **kwargs)
    return path
