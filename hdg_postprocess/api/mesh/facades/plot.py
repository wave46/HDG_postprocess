from hdg_postprocess.core.mesh import plotting as plotting_ops


class MeshPlot:
    def __init__(self, mesh):
        self._mesh = mesh

    def raw(self, data=None, ax=None):
        return plotting_ops.plot_raw_meshes(self._mesh, data=data, ax=ax)

    def full(
        self,
        data=None,
        ax=None,
        log=False,
        label=None,
        connectivity=None,
        n_levels=100,
        limits=None,
        ticks=None,
        tick_labels=None,
        cmap="jet",
        linewidth=1.0,
    ):
        return plotting_ops.plot_full_mesh(
            self._mesh,
            data=data,
            ax=ax,
            log=log,
            label=label,
            connectivity=connectivity,
            n_levels=n_levels,
            limits=limits,
            ticks=ticks,
            tick_labels=tick_labels,
            cmap=cmap,
            linewidth=linewidth,
        )

    def outline(self, raw_boundary_info=None, ax=None):
        return plotting_ops.plot_mesh_outline(self._mesh, raw_boundary_info=raw_boundary_info, ax=ax)

    def normals_tangentials(self, raw_boundary_info=None, ax=None, scale=None, scale_units=None):
        return plotting_ops.plot_mesh_normals_tangentials(
            self._mesh,
            raw_boundary_info=raw_boundary_info,
            ax=ax,
            scale=scale,
            scale_units=scale_units,
        )
