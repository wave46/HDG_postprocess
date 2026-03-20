import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm


DEFAULT_POSITIVE_CMAP = "magma"
DEFAULT_SIGNED_CMAP = "RdBu_r"


def _ensure_full_mesh(mesh):
    mesh.assembly.full()


def _ensure_boundary(mesh, raw_boundary_info):
    if raw_boundary_info is None:
        raise ValueError("Please, provide raw boundary info as input to this method")
    mesh.assembly.boundary(raw_boundary_info)


def _ensure_boundary_gauss(mesh, raw_boundary_info):
    if raw_boundary_info is None:
        raise ValueError("Please, provide raw boundary info as input to this method")
    unique_boundaries = tuple(np.unique(raw_boundary_info[0]["boundary_flags"]).tolist())
    mesh.assembly.boundary_gauss(unique_boundaries, raw_boundary_info)


def _default_plot_connectivity(mesh):
    if mesh.metadata.reference_element is None:
        print("No reference element, the mesh is plotted assuming straight edges")
        if mesh.mesh_parameters["element_type"] == "triangle":
            return mesh.global_state.connectivity[:, :3]
        return mesh.global_state.connectivity[:, :4]
    print("Full mesh is plotted includin curved edges")
    return mesh.global_state.connectivity[:, mesh.metadata.reference_element["faceNodes"].flatten()]


def _plot_with_connectivity(ax, vertices, connectivity, data, linewidth):
    if data is None:
        verts = vertices[connectivity]
        collection = PolyCollection(verts, facecolor="none", edgecolor="k", linewidth=linewidth)
        ax.add_collection(collection)
        return None
    if data.shape[0] == connectivity.shape[0]:
        verts = vertices[connectivity]
        collection = PolyCollection(verts)
        collection.set_array(data)
        ax.add_collection(collection)
        return None
    return "tricontourf"


def _make_triangulation(vertices, connectivity):
    return Triangulation(vertices[:, 0], vertices[:, 1], triangles=connectivity)


def _default_cmap(data, *, log, cmap):
    if cmap is not None:
        return cmap
    if log:
        return DEFAULT_POSITIVE_CMAP
    if data is None:
        return DEFAULT_POSITIVE_CMAP
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return DEFAULT_POSITIVE_CMAP
    if np.any(finite < 0) and np.any(finite > 0):
        return DEFAULT_SIGNED_CMAP
    return DEFAULT_POSITIVE_CMAP


def plot_raw_meshes(mesh, data=None, ax=None):
    colors = cm.get_cmap("hsv", mesh.n_partitions)
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    for i, (vertices, connectivity) in enumerate(zip(mesh.raw.vertices, mesh.raw.connectivity)):
        if mesh.metadata.reference_element is None:
            print("No reference element, the mesh is plotted assuming straight edges")
            if mesh.mesh_parameters["element_type"] == "triangle":
                verts = vertices[connectivity[:, :3]]
            else:
                verts = vertices[connectivity[:, :4]]
        else:
            print("Full mesh is plotted includin curved edges")
            verts = vertices[connectivity[:, mesh.metadata.reference_element["faceNodes"].flatten()]]

        if data is None:
            collection = PolyCollection(verts, facecolor="none", edgecolor=colors(i), linewidth=0.05)
        else:
            collection = PolyCollection(verts)
            collection.set_array(data[i][connectivity[:, :3]].mean(axis=1))
        ax.add_collection(collection)
    ax.set_aspect(1)
    ax.set_xlim(mesh.metadata.extent["minr"], mesh.metadata.extent["maxr"])
    ax.set_ylim(mesh.metadata.extent["minz"], mesh.metadata.extent["maxz"])
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    return ax


def plot_full_mesh(mesh, data=None, ax=None, log=False, label=None, connectivity=None,
                   n_levels=100, limits=None, ticks=None, tick_labels=None, cmap=None, linewidth=0.1):
    _ensure_full_mesh(mesh)
    cmap = _default_cmap(data, log=log, cmap=cmap)

    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    if connectivity is None:
        connectivity = _default_plot_connectivity(mesh)

    plot_mode = _plot_with_connectivity(ax, mesh.global_state.vertices, connectivity, data, linewidth)
    if plot_mode == "tricontourf":
        triangulation = _make_triangulation(mesh.global_state.vertices, connectivity)
        if log:
            if limits is None:
                im = ax.tricontourf(
                    triangulation,
                    np.log10(data),
                    levels=n_levels,
                    extend="both",
                    cmap=cmap,
                )
                ax.set_title(f"log10({label})")
            else:
                im = ax.tricontourf(
                    triangulation,
                    data,
                    levels=np.logspace(limits[0], limits[1], n_levels),
                    cmap=cmap, vmin=10.0 ** limits[0], vmax=10.0 ** limits[1],
                    norm=LogNorm(vmin=10.0 ** limits[0], vmax=10.0 ** limits[1]),
                    extend="both",
                )
                ax.set_title(f"{label}")
        else:
            if limits is None:
                im = ax.tricontourf(
                    triangulation,
                    data,
                    levels=n_levels,
                    extend="both",
                    cmap=cmap,
                )
            else:
                im = ax.tricontourf(
                    triangulation,
                    data,
                    levels=np.linspace(limits[0], limits[1], n_levels),
                    extend="both",
                    cmap=cmap,
                    vmin=limits[0],
                    vmax=limits[1],
                )
            ax.set_title(f"{label}")
        cbar = plt.colorbar(im, ax=ax, extendrect=True)
        if ticks is not None:
            cbar.set_ticks(ticks)
        if tick_labels is not None:
            cbar.set_ticklabels(tick_labels)

    ax.set_aspect(1)
    ax.set_xlim(mesh.metadata.extent["minr"], mesh.metadata.extent["maxr"])
    ax.set_ylim(mesh.metadata.extent["minz"], mesh.metadata.extent["maxz"])
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    return ax


def plot_mesh_outline(mesh, raw_boundary_info=None, ax=None):
    _ensure_full_mesh(mesh)
    _ensure_boundary(mesh, raw_boundary_info)
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    colors = ["r", "g", "b"]
    for i, (key, bound_connect) in enumerate(mesh.boundary_state.connectivity.items()):
        for k, single_connect in enumerate(bound_connect):
            vertices_boundary = mesh.global_state.vertices[single_connect]
            r = vertices_boundary[:, :, 0].flatten()
            z = vertices_boundary[:, :, 1].flatten()
            if k == 0:
                ax.plot(r, z, label=key, color=colors[i])
            else:
                ax.plot(r, z, color=colors[i])
    ax.set_aspect(1)
    ax.set_xlim(mesh.metadata.extent["minr"], mesh.metadata.extent["maxr"])
    ax.set_ylim(mesh.metadata.extent["minz"], mesh.metadata.extent["maxz"])
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.legend()
    return ax


def plot_mesh_normals_tangentials(mesh, raw_boundary_info=None, ax=None, scale=None, scale_units=None):
    _ensure_full_mesh(mesh)
    _ensure_boundary_gauss(mesh, raw_boundary_info)
    r = mesh.boundary_state.vertices_gauss[:, ::-1, 0].flatten()
    z = mesh.boundary_state.vertices_gauss[:, ::-1, 1].flatten()
    n_r = mesh.boundary_state.normals_gauss[:, ::-1, 0].flatten()
    n_z = mesh.boundary_state.normals_gauss[:, ::-1, 1].flatten()
    t_r = mesh.boundary_state.tangentials_gauss[:, ::-1, 0].flatten()
    t_z = mesh.boundary_state.tangentials_gauss[:, ::-1, 1].flatten()
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)

    if scale is None:
        scale = 1000
    if scale_units is None:
        scale_units = "xy"
    ax.quiver(r, z, n_r, n_z, scale=scale, scale_units=scale_units)
    ax.quiver(r, z, t_r, t_z, scale=scale, scale_units=scale_units, color="r")
    ax.set_aspect(1)
    ax.set_xlim(mesh.metadata.extent["minr"], mesh.metadata.extent["maxr"])
    ax.set_ylim(mesh.metadata.extent["minz"], mesh.metadata.extent["maxz"])
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    return ax
