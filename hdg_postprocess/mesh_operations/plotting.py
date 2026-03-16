import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm


def plot_raw_meshes(mesh, data=None, ax=None):
    colors = cm.get_cmap("hsv", mesh.n_partitions)
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    for i, (vertices, connectivity) in enumerate(zip(mesh.raw_vertices, mesh.raw_connectivity)):
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
    ax.set_xlim(mesh.mesh_extent["minr"], mesh.mesh_extent["maxr"])
    ax.set_ylim(mesh.mesh_extent["minz"], mesh.mesh_extent["maxz"])
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    return ax


def plot_full_mesh(mesh, data=None, ax=None, log=False, label=None, connectivity=None,
                   n_levels=100, limits=None, ticks=None, tick_labels=None, cmap="jet", linewidth=1.0):
    if not mesh.metadata.flags.combined_to_full:
        print("Comibining to full mesh")
        from hdg_postprocess.mesh_operations.geometry import recombine_full_mesh

        recombine_full_mesh(mesh)

    colors = "k"
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    if connectivity is None:
        connectivity = mesh.connectivity_glob[:, :3]
        if mesh.metadata.reference_element is None:
            print("No reference element, the mesh is plotted assuming straight edges")
            if mesh.mesh_parameters["element_type"] == "triangle":
                connectivity = mesh.connectivity_glob[:, :3]
            else:
                connectivity = mesh.connectivity_glob[:, :4]
        else:
            print("Full mesh is plotted includin curved edges")
            connectivity = mesh.connectivity_glob[:, mesh.metadata.reference_element["faceNodes"].flatten()]

    if data is None:
        verts = mesh.vertices_glob[connectivity]
        collection = PolyCollection(verts, facecolor="none", edgecolor=colors, linewidth=linewidth)
        ax.add_collection(collection)
    else:
        if data.shape[0] == connectivity.shape[0]:
            verts = mesh.vertices_glob[connectivity]
            collection = PolyCollection(verts)
            collection.set_array(data)
            ax.add_collection(collection)
        else:
            if log:
                if limits is None:
                    im = ax.tricontourf(
                        mesh.vertices_glob[:, 0], mesh.vertices_glob[:, 1], np.log10(data),
                        levels=n_levels, extend="both", triangles=connectivity, cmap=cmap, extendrect=True,
                    )
                    ax.set_title(f"log10({label})")
                else:
                    im = ax.tricontourf(
                        mesh.vertices_glob[:, 0], mesh.vertices_glob[:, 1], data,
                        levels=np.logspace(limits[0], limits[1], n_levels), triangles=connectivity,
                        cmap=cmap, vmin=10.0 ** limits[0], vmax=10.0 ** limits[1],
                        norm=LogNorm(vmin=10.0 ** limits[0], vmax=10.0 ** limits[1]),
                        extend="both", extendrect=True,
                    )
                    ax.set_title(f"{label}")
            else:
                if limits is None:
                    im = ax.tricontourf(
                        mesh.vertices_glob[:, 0], mesh.vertices_glob[:, 1], data,
                        levels=n_levels, extend="both", triangles=connectivity, cmap=cmap, extendrect=True,
                    )
                else:
                    im = ax.tricontourf(
                        mesh.vertices_glob[:, 0], mesh.vertices_glob[:, 1], data,
                        levels=np.linspace(limits[0], limits[1], n_levels), extend="both",
                        triangles=connectivity, cmap=cmap, vmin=limits[0], vmax=limits[1], extendrect=True,
                    )
                ax.set_title(f"{label}")
            cbar = plt.colorbar(im, ax=ax, extendrect=True)
            if ticks is not None:
                cbar.set_ticks(ticks)
            if tick_labels is not None:
                cbar.set_ticklabels(tick_labels)

    ax.set_aspect(1)
    ax.set_xlim(mesh.mesh_extent["minr"], mesh.mesh_extent["maxr"])
    ax.set_ylim(mesh.mesh_extent["minz"], mesh.mesh_extent["maxz"])
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    return ax


def plot_mesh_outline(mesh, raw_boundary_info=None, ax=None):
    if not mesh.metadata.flags.combined_to_full:
        print("Comibining to full mesh")
        from hdg_postprocess.mesh_operations.geometry import recombine_full_mesh

        recombine_full_mesh(mesh)
    if not mesh.metadata.flags.boundary_combined:
        if raw_boundary_info is None:
            raise ValueError("Please, provide raw boundary info as input to this method")
        print("Comibining boundary")
        from hdg_postprocess.mesh_operations.boundary import recombine_full_boundary

        recombine_full_boundary(mesh, raw_boundary_info)
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    colors = ["r", "g", "b"]
    for i, (key, bound_connect) in enumerate(mesh.boundary_state.connectivity.items()):
        for k, single_connect in enumerate(bound_connect):
            vertices_boundary = mesh.vertices_glob[single_connect]
            r = vertices_boundary[:, :, 0].flatten()
            z = vertices_boundary[:, :, 1].flatten()
            if k == 0:
                ax.plot(r, z, label=key, color=colors[i])
            else:
                ax.plot(r, z, color=colors[i])
    ax.set_aspect(1)
    ax.set_xlim(mesh.mesh_extent["minr"], mesh.mesh_extent["maxr"])
    ax.set_ylim(mesh.mesh_extent["minz"], mesh.mesh_extent["maxz"])
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.legend()
    return ax


def plot_mesh_normals_tangentials(mesh, raw_boundary_info=None, ax=None, scale=None, scale_units=None):
    if not mesh.metadata.flags.combined_to_full:
        print("Comibining to full mesh")
        from hdg_postprocess.mesh_operations.geometry import recombine_full_mesh

        recombine_full_mesh(mesh)
    if not mesh.metadata.flags.boundary_gauss_initialized:
        if raw_boundary_info is None:
            raise ValueError("Please, provide raw boundary info as input to this method")
        print("Calculating at gauss points")
        from hdg_postprocess.mesh_operations.boundary import calculate_gauss_boundary

        unique_boundaries = tuple(np.unique(raw_boundary_info[0]["boundary_flags"]).tolist())
        calculate_gauss_boundary(mesh, unique_boundaries, raw_boundary_info)
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
    ax.set_xlim(mesh.mesh_extent["minr"], mesh.mesh_extent["maxr"])
    ax.set_ylim(mesh.mesh_extent["minz"], mesh.mesh_extent["maxz"])
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    return ax
