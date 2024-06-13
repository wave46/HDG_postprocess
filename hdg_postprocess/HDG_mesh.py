import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from raysect.core.math.function.float import Discrete2DMesh
from pathlib import Path
from matplotlib.colors import LogNorm
import os


class HDGmesh:
    """
    SOLEDGE-HDG mesh object
    The mesh is triangular (so far), high order  (so far p=4 or p=6), so it can have more than 3 nodes per element
    """

    def __init__(
        self,
        raw_vertices,
        raw_connectivity,
        raw_connectivity_boundary,
        raw_mesh_numbers,
        raw_boundary_flags,
        raw_ghost_elements,
        raw_ghost_faces,
        mesh_parameters,
        n_partitions,
        raw_rest_mesh_data=None,
    ):
        """
        Here we just store the raw data
        """
        self._raw_vertices = raw_vertices
        self._raw_connectivity = raw_connectivity
        self._raw_connectivity_boundary = raw_connectivity_boundary
        self._raw_mesh_numbers = raw_mesh_numbers
        self._raw_boundary_flags = raw_boundary_flags
        self._raw_ghost_elements = raw_ghost_elements
        self._raw_ghost_faces = raw_ghost_faces
        self._mesh_parameters = mesh_parameters
        self._n_partitions = n_partitions
        if n_partitions > 1:
            if raw_rest_mesh_data is None:
                raise ValueError("communication info is not provided")

            self._raw_rest_mesh_data = raw_rest_mesh_data

        self._initial_setup()

    def _initial_setup(self):

        if self.mesh_parameters["element_type"] == "triangle":
            if self.mesh_parameters["nodes_per_element"] == 15:
                self._p_order = 4
        elif self.mesh_parameters["element_type"] == "quadrilateral":
            if self.mesh_parameters["nodes_per_element"] == 49:
                self._p_order = 6
            elif self.mesh_parameters["nodes_per_element"] == 81:
                self._p_order = 8

        minr, maxr, minz, maxz = 1e5, -1e5, 1e5, -1e5
        for vertices in self.raw_vertices:

            minr = min(minr, vertices[:, 0].min())
            minz = min(minz, vertices[:, 1].min())
            maxr = max(maxr, vertices[:, 0].max())
            maxz = max(maxz, vertices[:, 1].max())
        self._mesh_extent = {"minr": minr, "maxr": maxr, "minz": minz, "maxz": maxz}
        self._connectivity_big = None
        self._element_number = None
        self._reference_element = None
        self._vertices_gauss = None
        self._vertices_boundary_gauss = None
        self._volumes_gauss = None
        self._tangentials_gauss = None
        self._normal_guass = None
        self._segment_length_gauss = None
        self._segment_surface_gauss = None
        # not sure if this will be used for serial version
        self._filled = None
        self._indices = None
        self._face_element_number = None
        self._face_local_number = None
        self._face_ghost = None
        if self._n_partitions == 1:
            # no need to combine meshes
            self._combined_to_full = True
            self._connectivity_glob = self.raw_connectivity[0]
            self._connectivity_b_glob = None
            self._vertices_glob = self.raw_vertices[0]
            self._nelems_glob = self._connectivity_glob.shape[0]
            self._nvertices_glob = self._vertices_glob.shape[0]
            self._nfaces_glob = None
            self._boundary_combined = False

        else:
            self._combined_to_full = False
            self._connectivity_glob = None
            self._connectivity_b_glob = None
            self._boundary_flags = None
            self._vertices_glob = None
            self._nelems_glob = None
            self._nvertices_glob = None
            self._nfaces_glob = None
            self._boundary_combined = False

    @property
    def raw_vertices(self):
        """raw vertices"""
        return self._raw_vertices

    @property
    def raw_connectivity(self):
        """raw connectivity"""
        return self._raw_connectivity

    @property
    def vertices_glob(self):
        """vertices in global mesh"""
        return self._vertices_glob

    @property
    def vertices_gauss(self):
        """coordinates of gauss points in global mesh"""
        return self._vertices_gauss

    @property
    def vertices_boundary_gauss(self):
        """coordinates of gauss points in global mesh"""
        return self._vertices_boundary_gauss

    @property
    def tangentials_gauss(self):
        """tangentials to mesh boundary at gauss points in global mesh"""
        return self._tangentials_gauss

    @property
    def normals_gauss(self):
        """normals to mesh boundary at gauss points in global mesh"""
        return self._normals_gauss

    @property
    def segment_length_gauss(self):
        """segment length corresponding to each gauss point on the boundary"""
        return self._segment_length_gauss

    @property
    def segment_surface_gauss(self):
        """segment length corresponding to each gauss point on the boundary"""
        return self._segment_surface_gauss

    @property
    def volumes_gauss(self):
        """volumes in gauss points in global mesh"""
        return self._volumes_gauss

    @property
    def connectivity_glob(self):
        """connectivity of a global mesh"""
        return self._connectivity_glob

    @property
    def connectivity_b_glob(self):
        """connectivity of faces on global mesh. filled only with boundary faces"""
        return self._connectivity_b_glob

    @property
    def boundary_flags(self):
        """
        faces flags on a full mesh.
        -1 means that it is inside boundary, not filled int
        0 boundary between two partitions
        to be done: fill other flags
        """
        return self._boundary_flags

    @property
    def face_element_number(self):
        """
        for each face gives a number of the corresponding element
        """
        return self._face_element_number

    @property
    def face_local_number(self):
        """
        for each face gives a local number of the face in corresponding element
        """
        return self._face_local_number

    @property
    def face_ghost(self):
        """
        If face corresponds to ghost element
        """
        return self._face_ghost

    @property
    def connectivity_big(self):
        """big connectivity of a global mesh for plots
        (each element is triangulated)
        """
        return self._connectivity_big

    @property
    def nelems_glob(self):
        """number of elements in global mesh"""
        return self._nelems_glob

    @property
    def nvertices_glob(self):
        """nubmer of vertices in global mesh"""
        return self._nvertices_glob

    @property
    def nfaces_glob(self):
        """number of faces in global mesh"""
        return self._nfaces_glob

    @property
    def combined_to_full(self):
        """Flag which tells if the mesh has been combined to full"""
        return self._combined_to_full

    @property
    def raw_connectivity_boundary(self):
        """raw connectivity at the boundary"""
        return self._raw_connectivity_boundary

    @property
    def raw_mesh_numbers(self):
        """raw mesh numbers dictionary"""
        return self._raw_mesh_numbers

    @property
    def raw_boundary_flags(self):
        """raw mesh boundary flags"""
        return self._raw_boundary_flags

    @property
    def raw_ghost_elements(self):
        """raw mesh ghost elements flags"""
        return self._raw_ghost_elements

    @property
    def raw_ghost_faces(self):
        """raw mesh ghost elements flags"""
        return self._raw_ghost_faces

    @property
    def raw_rest_mesh_data(self):
        """raw rest mesh data"""
        return self._raw_rest_mesh_data

    @property
    def mesh_parameters(self):
        """mesh parameters"""
        return self._mesh_parameters

    @property
    def n_partitions(self):
        """number of partitions"""
        return self._n_partitions

    @property
    def p_order(self):
        """polynomial order of the mesh"""
        return self._p_order

    @property
    def mesh_extent(self):
        """Extent of the mesh. A dictionary with minr, maxr, minz and maxz keys."""
        return self._mesh_extent

    @property
    def mask(self):
        """Mesh mask which gives 1 if"""
        return self._mask

    @property
    def element_number(self):
        """
        For given pair (R,Z) gives a number of element to which this point relates
        Outside of the mesh gives -1
        """
        return self._element_number

    @element_number.setter
    def element_number(self, value):
        self._element_number = value

    @property
    def boundary_combined(self):
        """Flag which tells if the mesh has been combined on the boundary"""
        return self._boundary_combined

    @property
    def reference_element(self):
        """Dictionary with atomic parameters"""
        return self._reference_element

    @reference_element.setter
    def reference_element(self, value):
        self._reference_element = value

    def plot_raw_meshes(self, data=None, ax=None):
        """
        Plot all raw meshes to a matplotlib figure.
        :param data: Data array defined on the soledgehdg mesh
        """
        colors = cm.get_cmap("hsv", self.n_partitions)
        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)
        for i, (vertices, connectivity) in enumerate(
            zip(self.raw_vertices, self.raw_connectivity)
        ):
            if self.reference_element is None:
                print(
                    "No reference element, the mesh is plotted assuming straight edges"
                )
                if self.mesh_parameters["element_type"] == "triangle":
                    verts = vertices[connectivity[:, :3]]
                else:
                    verts = vertices[connectivity[:, :4]]
            else:
                print("Full mesh is plotted includin curved edges")
                verts = vertices[
                    connectivity[:, self.reference_element["faceNodes"].flatten()]
                ]

            if data is None:

                collection = PolyCollection(
                    verts, facecolor="none", edgecolor=colors(i), linewidth=0.05
                )

            else:
                collection = PolyCollection(verts)
                collection.set_array(data[i][connectivity[:, :3]].mean(axis=1))
            ax.add_collection(collection)
        ax.set_aspect(1)
        ax.set_xlim(self.mesh_extent["minr"], self.mesh_extent["maxr"])
        ax.set_ylim(self.mesh_extent["minz"], self.mesh_extent["maxz"])
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        return ax

    def plot_full_mesh(
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
    ):
        """
        Plot all raw meshes to a matplotlib figure.
        :param data: Data array defined on the soledgehdg mesh
        :param ax: ax where to plot array defined on the soledgehdg mesh
        :param log: if plot in log scale
        :param label: label for the variable
        :param connectivity: if None use the global one, (maybe refined, i.e. each triangle is also triangulated)
        """
        if not self._combined_to_full:

            print("Comibining to full mesh")
            self.recombine_full_mesh()

        colors = "b"
        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)
        if connectivity is None:
            connectivity = self.connectivity_glob[:, :3]
            if self.reference_element is None:
                print(
                    "No reference element, the mesh is plotted assuming straight edges"
                )
                if self.mesh_parameters["element_type"] == "triangle":
                    connectivity = self.connectivity_glob[:, :3]
                else:
                    connectivity = self.connectivity_glob[:, :4]
            else:
                print("Full mesh is plotted includin curved edges")
                connectivity = self.connectivity_glob[
                    :, self.reference_element["faceNodes"].flatten()
                ]

        if data is None:
            verts = self.vertices_glob[connectivity]
            collection = PolyCollection(
                verts, facecolor="none", edgecolor=colors, linewidth=0.05
            )
            ax.add_collection(collection)
        else:
            if data.shape[0] == connectivity.shape[0]:
                verts = self.vertices_glob[connectivity]
                collection = PolyCollection(verts)
                collection.set_array(data)
                ax.add_collection(collection)
            else:

                if log:
                    if limits == None:
                        im = ax.tricontourf(
                            self.vertices_glob[:, 0],
                            self.vertices_glob[:, 1],
                            np.log10(data),
                            levels=n_levels,
                            extend="both",
                            triangles=connectivity,
                            cmap="jet",
                            extendrect=True,
                        )
                        ax.set_title(f"log10({label})")
                    else:
                        im = ax.tricontourf(
                            self.vertices_glob[:, 0],
                            self.vertices_glob[:, 1],
                            data,
                            levels=np.logspace(limits[0], limits[1], n_levels),
                            triangles=connectivity,
                            cmap="jet",
                            vmin=10.0 ** limits[0],
                            vmax=10.0 ** limits[1],
                            norm=LogNorm(
                                vmin=10.0 ** limits[0], vmax=10.0 ** limits[1]
                            ),
                            extend="both",
                            extendrect=True,
                        )

                    ax.set_title(f"{label}")
                else:
                    if limits == None:
                        im = ax.tricontourf(
                            self.vertices_glob[:, 0],
                            self.vertices_glob[:, 1],
                            data,
                            levels=n_levels,
                            extend="both",
                            triangles=connectivity,
                            cmap="jet",
                            extendrect=True,
                        )
                    else:
                        im = ax.tricontourf(
                            self.vertices_glob[:, 0],
                            self.vertices_glob[:, 1],
                            data,
                            levels=np.linspace(limits[0], limits[1], n_levels),
                            extend="both",
                            triangles=connectivity,
                            cmap="jet",
                            vmin=limits[0],
                            vmax=limits[1],
                            extendrect=True,
                        )
                    ax.set_title(f"{label}")
                cbar = plt.colorbar(im, ax=ax, extendrect=True)
                if ticks is not None:

                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels(tick_labels)

        ax.set_aspect(1)
        ax.set_xlim(self.mesh_extent["minr"], self.mesh_extent["maxr"])
        ax.set_ylim(self.mesh_extent["minz"], self.mesh_extent["maxz"])
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        return ax

    def plot_mesh_outline(self, raw_boundary_info=None, ax=None):
        """
        Plot mesh outline
        :param raw_boundary_info: is the list of dictio naries with additional boundary info which is saved in solution
        :param ax: ax where to plot
        """
        if not self._combined_to_full:
            print("Comibining to full mesh")
            self.recombine_full_mesh()
        if not self._boundary_combined:
            if raw_boundary_info is None:
                raise ValueError(
                    "Please, provide raw boundary info as input to this method"
                )
            print("Comibining boundary")
            self.recombine_full_boundary(raw_boundary_info)
        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)
        colors = ["r", "g", "b"]
        for i, (key, bound_connect) in enumerate(self.connectivity_b_glob.items()):
            for k, single_connect in enumerate(bound_connect):
                vertices_boundary = self.vertices_glob[single_connect]
                r = vertices_boundary[:, :, 0].flatten()
                z = vertices_boundary[:, :, 1].flatten()
                if k == 0:
                    ax.plot(r, z, label=key, color=colors[i])
                else:
                    ax.plot(r, z, color=colors[i])
        ax.set_aspect(1)
        ax.set_xlim(self.mesh_extent["minr"], self.mesh_extent["maxr"])
        ax.set_ylim(self.mesh_extent["minz"], self.mesh_extent["maxz"])
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.legend()
        return ax

    def plot_mesh_normals_tangentials(
        self, raw_boundary_info=None, ax=None, scale=None, scale_units=None
    ):
        """
        Plots quiver plot of tangent and normal vectors to the mesh boundary at the gauss points
        :param raw_boundary_info: is the list of dictio naries with additional boundary info which is saved in solution
        :param ax: ax where to plot
        :param scale: same meaning as in plt.quiver
        :param scale_units: same meaning as in plt.quiver
        by default the size of tangential and normal vectors is normalized to 1 mm on the plot
        It may be adjusted using scale and scale_units settings
        """
        if not self._combined_to_full:

            print("Comibining to full mesh")
            self.recombine_full_mesh()
        if self._vertices_boundary_gauss is None:
            if raw_boundary_info is None:
                raise ValueError(
                    "Please, provide raw boundary info as input to this method"
                )
            print("Calculating at gauss points")
            self.calculate_gauss_boundary(raw_boundary_info)
        # important to inverse second axis,because gauss points are in reverse order
        r = self.vertices_boundary_gauss[:, ::-1, 0].flatten()
        z = self.vertices_boundary_gauss[:, ::-1, 1].flatten()
        # normals
        n_r = self.normals_gauss[:, ::-1, 0].flatten()
        n_z = self.normals_gauss[:, ::-1, 1].flatten()
        # tangents
        t_r = self.tangentials_gauss[:, ::-1, 0].flatten()
        t_z = self.tangentials_gauss[:, ::-1, 1].flatten()
        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        if scale is None:
            scale = 1000
        if scale_units is None:
            scale_units = "xy"
        ax.quiver(r, z, n_r, n_z, scale=scale, scale_units=scale_units)
        ax.quiver(r, z, t_r, t_z, scale=scale, scale_units=scale_units, color="r")
        ax.set_aspect(1)
        ax.set_xlim(self.mesh_extent["minr"], self.mesh_extent["maxr"])
        ax.set_ylim(self.mesh_extent["minz"], self.mesh_extent["maxz"])
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        return ax

    def recombine_full_mesh(self):

        self._nelems_glob = 0
        self._nvertices_glob = 0

        for i in range(self.n_partitions):
            self._nelems_glob = max(
                self._nelems_glob, self.raw_rest_mesh_data[i]["loc2glob_el"].max()
            )
            self._nvertices_glob = max(
                self._nvertices_glob, self.raw_rest_mesh_data[i]["loc2glob_no"].max()
            )
        self._nelems_glob += 1
        self._nvertices_glob += 1
        self._connectivity_glob = np.zeros(
            (self._nelems_glob, self.mesh_parameters["nodes_per_element"]), dtype=int
        )
        self._vertices_glob = np.zeros(
            (self._nvertices_glob, self.mesh_parameters["Ndim"])
        )
        for i in range(self.n_partitions):
            # removing ghost elements as well
            self._connectivity_glob[
                self.raw_rest_mesh_data[i]["loc2glob_el"][
                    ~self.raw_ghost_elements[i].astype(bool).flatten()
                ]
            ] = self.raw_rest_mesh_data[i]["loc2glob_no"][
                self.raw_connectivity[i][
                    ~self.raw_ghost_elements[i].astype(bool).flatten(), :
                ]
            ]
            self._vertices_glob[self.raw_rest_mesh_data[i]["loc2glob_no"], :] = (
                self.raw_vertices[i][:, :]
            )
        self._combined_to_full = True

    def recombine_full_boundary(self, raw_boundary_info):
        """
        Recombines full boundary connectivity with all needed information to calculate fluxes at the wall
        """
        if self.n_partitions > 1:
            self._nfaces_glob = 0
            for i in range(self.n_partitions):
                self._nfaces_glob = max(
                    self._nfaces_glob, self.raw_rest_mesh_data[i]["loc2glob_fa"].max()
                )
            self._nfaces_glob += 1
            connectivity_b_glob = -1 * np.ones(
                (self._nfaces_glob, self.mesh_parameters["nodes_per_face"]), dtype=int
            )
            boundary_flags = -1 * np.ones((self._nfaces_glob), dtype=int)
            face_element_number = np.zeros((self._nfaces_glob, 1), dtype=int)
            face_local_number = np.zeros((self._nfaces_glob), dtype=int)

            for i in range(self.n_partitions):
                non_ghost = (~self.raw_ghost_faces[i].flatten())[
                    -self.raw_mesh_numbers[i]["Nextfaces"] :
                ]

                connectivity_b_glob[
                    self.raw_rest_mesh_data[i]["loc2glob_fa"][
                        -self.raw_mesh_numbers[i]["Nextfaces"] :
                    ][non_ghost],
                    :,
                ] = self.raw_rest_mesh_data[i]["loc2glob_no"][
                    self.raw_connectivity_boundary[i][:, :]
                ][
                    non_ghost
                ]
                boundary_flags[
                    self.raw_rest_mesh_data[i]["loc2glob_fa"][
                        -self.raw_mesh_numbers[i]["Nextfaces"] :
                    ][non_ghost]
                ] = raw_boundary_info[i]["boundary_flags"][non_ghost]
                face_element_number[
                    self.raw_rest_mesh_data[i]["loc2glob_fa"][
                        -self.raw_mesh_numbers[i]["Nextfaces"] :
                    ][non_ghost]
                ] = self.raw_rest_mesh_data[i]["loc2glob_el"][
                    raw_boundary_info[i]["exterior_faces"][:, 0][non_ghost]
                ][
                    None
                ].T
                face_local_number[
                    self.raw_rest_mesh_data[i]["loc2glob_fa"][
                        -self.raw_mesh_numbers[i]["Nextfaces"] :
                    ][non_ghost]
                ] = raw_boundary_info[i]["exterior_faces"][:, 1][non_ghost]

            # we filled in only exterior faces info
            # excluding empty entrances
            filled = (connectivity_b_glob != -1).all(axis=1)
            connectivity_b_glob = connectivity_b_glob[filled, :]
            boundary_flags = boundary_flags[filled]
            face_element_number = face_element_number[filled]
            face_local_number = face_local_number[filled]

            # save to make easier skeleton solution rebuilding
            self._filled = filled
        else:
            self._nfaces_glob = self._raw_connectivity_boundary[0].shape[0]
            connectivity_b_glob = self._raw_connectivity_boundary[0]
            boundary_flags = raw_boundary_info[0]["boundary_flags"]
            face_element_number = raw_boundary_info[0]["exterior_faces"][:, 0][None].T
            face_local_number = raw_boundary_info[0]["exterior_faces"][:, 1]
        # this boundary is not ordered, so now we reorder it
        # also the boundary is splitted according to boundary flags and saved in dictionary
        unique_boundaries = np.unique(boundary_flags)
        indices = {}
        self._connectivity_b_glob = {}
        self._boundary_flags = {}
        self._face_element_number = {}
        self._face_local_number = {}
        for boundary_type in unique_boundaries:
            # choosing this boundary
            boundary_idx = np.where(boundary_type == boundary_flags)[0]
            bound_connectivity = connectivity_b_glob[boundary_idx, :]
            # now choosing the index to start mesh recombining
            starting_indices = bound_connectivity[:, 0]
            ending_indices = bound_connectivity[:, -1]
            difference = np.setdiff1d(starting_indices, ending_indices)
            if len(difference) == 0:
                # this means that the border is closed and continous
                all_ind = []
                ind = [boundary_idx[0]]
                i = 0
                while len(ind) != len(bound_connectivity):

                    i = np.where(bound_connectivity[i, -1] == bound_connectivity[:, 0])[
                        0
                    ][0]
                    ind.append(boundary_idx[i])
                all_ind.append(ind)

            else:
                # We start with first segment and if there are more, we take next index in difference
                k = 0
                starting_ind = difference[k]
                i = np.where(starting_ind == starting_indices)[0][0]
                all_ind = []
                ind = [boundary_idx[i]]
                recorded_indexes = 1
                while recorded_indexes != len(bound_connectivity):
                    i = np.where(bound_connectivity[i, -1] == bound_connectivity[:, 0])[
                        0
                    ]
                    if len(i) != 0:
                        i = i[0]
                        ind.append(boundary_idx[i])
                    else:
                        all_ind.append(ind)
                        k += 1
                        starting_ind = difference[k]
                        i = np.where(starting_ind == starting_indices)[0][0]
                        ind = [boundary_idx[i]]

                    recorded_indexes += 1
                all_ind.append(ind)
            self._connectivity_b_glob[boundary_type] = []
            self._face_element_number[boundary_type] = []
            self._face_local_number[boundary_type] = []
            self._boundary_flags[boundary_type] = []
            for ind in all_ind:

                self._connectivity_b_glob[boundary_type].append(
                    connectivity_b_glob[ind, :]
                )
                self._boundary_flags[boundary_type].append(boundary_flags[ind])
                self._face_element_number[boundary_type].append(
                    face_element_number[ind]
                )
                self._face_local_number[boundary_type].append(face_local_number[ind])
            indices[boundary_type] = all_ind

        # save to make easier skeleton solution rebuilding
        self._indices = indices
        self._boundary_combined = True

    def boundary_ordering(self, raw_boundary_info, boundaries):
        """
        Recombines ordered boundary connectivity for given boundaries with all needed information to calculate fluxes at the wall
        """

        if not self._boundary_combined:
            self.recombine_full_boundary(raw_boundary_info)

        connected_boundaries = []
        segments = 0
        for boundary in boundaries:
            connected_boundary = []
            for sub_connectivity in self.connectivity_b_glob[boundary]:
                connected_boundary.append(False)
                segments += 1
            connected_boundaries.append(connected_boundary)
        boundary_ordering = [[0, 0]]
        connected_boundaries[0][0] = True
        segments -= 1
        node_idx = self.connectivity_b_glob[boundaries[0]][0][-1, -1]
        while segments > 0:
            for i, connected_boundary in enumerate(connected_boundaries):
                for j, segment in enumerate(connected_boundary):
                    if not segment:
                        if node_idx == self.connectivity_b_glob[boundaries[i]][j][0, 0]:
                            connected_boundaries[i][j] == True
                            segments -= 1
                            boundary_ordering.append([i, j])
                            node_idx = self.connectivity_b_glob[boundaries[i]][j][
                                -1, -1
                            ]

        connectivity_b_ordered = np.empty(
            [0, self.connectivity_b_glob[boundaries[0]][0].shape[1]], dtype=int
        )
        for bound_ordering in boundary_ordering:
            connectivity_b_ordered = np.vstack(
                [
                    connectivity_b_ordered,
                    self.connectivity_b_glob[boundaries[bound_ordering[0]]][
                        bound_ordering[1]
                    ],
                ]
            )

        return boundary_ordering, connectivity_b_ordered

    def create_connectivity_big(self):
        # additional triangulaton: since we have more than 3 points in each element, we can triangulate it
        # take any triangle from the mesh
        if not self._combined_to_full:
            self.recombine_full_mesh()
        # defining triangulation order of each big triangle
        base_path = Path(__file__).parent
        rel_path = f'data/triangulations_element/{self.mesh_parameters["element_type"]}_P{self.p_order}.npy'
        path = (base_path / rel_path).resolve()
        if not os.path.isfile(path):
            raise KeyError(f"{path} splitting of each element is not defined yet")

        triangle_indexes = np.load(path)

        connectivity_big = self.connectivity_glob[:, triangle_indexes]
        self._connectivity_big = connectivity_big.reshape(
            connectivity_big.shape[0] * connectivity_big.shape[1], 3
        )

    def make_mask(self):
        """
        to do create interpolator using
        """
        if self.connectivity_big is None:
            self.create_connectivity_big()

        self._mask = Discrete2DMesh(
            self.vertices_glob,
            self.connectivity_big,
            np.ones(self.connectivity_big.shape[0]),
            limit=False,
            default_value=0,
        )

    def make_element_number_funtion(self):
        """
        creates a function which gives number of element for givern point
        if outside of the mesh, it gives -1
        """
        if self.connectivity_big is None:
            self.create_connectivity_big()
        element_numbers = np.repeat(
            np.arange(len(self.connectivity_glob)),
            self.connectivity_big.shape[0] / self.connectivity_glob.shape[0],
        )
        self._element_number = Discrete2DMesh(
            self.vertices_glob,
            self.connectivity_big,
            element_numbers,
            limit=False,
            default_value=-1,
        )

    def calculate_gauss_volumes(self):
        """
        calculates volumes for each gauss point in the full mesh
        this is neede for volume integration later
        """
        if self.reference_element is None:
            raise ValueError("Please, provide reference element")
        if not self._combined_to_full:
            self.recombine_full_mesh()

        self._vertices_gauss = np.einsum(
            "ij,kjh->kih",
            self.reference_element["N"],
            self.vertices_glob[self.connectivity_glob, :],
        )
        J11_loc = np.einsum(
            "ij,kj->ki",
            self.reference_element["Nxi"],
            self.vertices_glob[self.connectivity_glob, 0],
        )
        J12_loc = np.einsum(
            "ij,kj->ki",
            self.reference_element["Nxi"],
            self.vertices_glob[self.connectivity_glob, 1],
        )
        J21_loc = np.einsum(
            "ij,kj->ki",
            self.reference_element["Neta"],
            self.vertices_glob[self.connectivity_glob, 0],
        )
        J22_loc = np.einsum(
            "ij,kj->ki",
            self.reference_element["Neta"],
            self.vertices_glob[self.connectivity_glob, 1],
        )
        detJ_loc = J11_loc * J22_loc - J12_loc * J21_loc

        self._volumes_gauss = (
            2
            * np.pi
            * self.reference_element["IPweights"][None, :]
            * detJ_loc[:, :]
            * self._vertices_gauss[:, :, 0]
        )

    def calculate_gauss_boundary(self, boundaries, raw_boundary_info):
        """
        calculates:
        vertices in gauss points for boundary faces
        tangential and normal vectors in each point
        segment lengths corresponding to the points
        """
        if self.reference_element is None:
            raise ValueError("Please, provide reference element")
        if not self._combined_to_full:
            self.recombine_full_mesh()
        if not self._boundary_combined:
            if raw_boundary_info is None:
                raise ValueError(
                    "Please, provide raw boundary info as input to this method"
                )
            self.recombine_full_boundary(raw_boundary_info)
        boundary_ordering, connectivity_ordered = self.boundary_ordering(
            raw_boundary_info, boundaries
        )

        self._vertices_boundary_gauss = np.einsum(
            "ij,kjh->kih",
            self.reference_element["N1d"],
            self.vertices_glob[connectivity_ordered, :],
        )

        # shape functions derivative at gauss points
        derivative_gauss = np.einsum(
            "ij,kjh->kih",
            self.reference_element["N1dxi"],
            self.vertices_glob[connectivity_ordered, :],
        )
        derivative_norm = np.sqrt(((derivative_gauss**2).sum(axis=2)))[:, :, None]
        self._tangentials_gauss = derivative_gauss / derivative_norm
        self._normals_gauss = np.zeros_like(self._tangentials_gauss)
        self._normals_gauss[:, :, 0] = self._tangentials_gauss[:, :, 1]
        self._normals_gauss[:, :, 1] = -1 * self._tangentials_gauss[:, :, 0]
        # todo: if non axy-symmetric
        self._segment_length_gauss = (
            derivative_norm * self.reference_element["IPweights1d"][None, :, :]
        )
        self._segment_surface_gauss = (
            self._segment_length_gauss
            * 2
            * np.pi
            * self._vertices_boundary_gauss[:, :, 0][:, :, None]
        )

        return boundary_ordering, connectivity_ordered

    def find_adjacent_elements(self, element_number):
        """
        finds numbers of adjacent elements
        """
        if not self._combined_to_full:
            print("Comibining to full mesh")
            self.recombine_full_mesh()
        vertices_numbers = self.connectivity_glob[element_number]

        adjacent_numbers = []
        for number in vertices_numbers:
            idx = np.where(number == self.connectivity_glob)
            for i in idx[0]:
                if (i != element_number) and (i not in adjacent_numbers):
                    adjacent_numbers.append(i)
        return adjacent_numbers
