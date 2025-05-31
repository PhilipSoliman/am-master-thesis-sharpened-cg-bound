import copy
import json
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import netgen.libngpy._NgOCC as occ
import ngsolve as ngs
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon

from lib.utils import CUSTOM_COLORS_SIMPLE, get_root

DATA_DIR = get_root() / "data"


def OCCRectangle(l, w):
    return occ.WorkPlane().Rectangle(l, w)


class BoundaryName(Enum):
    """
    Enum for boundary names used in the TwoLevelMesh class.
    """

    BOTTOM = "bottom"
    RIGHT = "right"
    TOP = "top"
    LEFT = "left"


class TwoLevelMesh:
    """
    TwoLevelMesh
    A class for generating, managing, and visualizing two-level conforming meshes (coarse and fine) for rectangular domains, with support for layersping domain decomposition and mesh serialization.
    Attributes:
        lx (float): Length of the domain in the x-direction.
        ly (float): Length of the domain in the y-direction.
        coarse_mesh_size (float): Mesh size for the coarse mesh.
        refinement_levels (int): Number of uniform refinements for the fine mesh.
        fine_mesh (ngs.Mesh): Refined fine mesh.
        coarse_mesh (ngs.Mesh): Coarse mesh.
        subdomains (dict): Mapping of coarse elements to their corresponding fine mesh domains and layers.
        layers(int): Number of layers for domain decomposition.
    Methods:
        - __init__(lx, ly, coarse_mesh_size, refinement_levels=1, layers=0):
            Initialize the TwoLevelMesh with domain size, mesh size, refinement, and layers.
        - create_conforming_meshes():
            Create conforming coarse and fine meshes for the rectangular domain.
        - get_subdomains():
            Identify and return the mapping of coarse mesh elements to their corresponding fine mesh domains.
        - get_subdomain_domain_edges(subdomain, interior_elements):
            Find all fine mesh edges that lie on the edges of a given coarse mesh element.
        - extend_subdomains(layer_idx=1):
            Extend the subdomains by adding layers of fine mesh elements.
        - save(file_name=""):
            Save the mesh, metadata, and subdomain information to disk.
        - load(file_name=""):
            Class method to load a TwoLevelMesh instance from saved files.
        - plot_domains(domains=1, plot_layers=False, opacity=0.9, fade_factor=1.5):
            Visualize the subdomains and optional layers using matplotlib.
        - plot_element(ax, element, mesh, fillcolor="blue", edgecolor="black", alpha=1.0, label=None):
            Static method to plot a single mesh element on a matplotlib axis.
    Usage:
        - Construct a TwoLevelMesh for a rectangular domain.
        - Save and load mesh and domain decomposition data.
        - Visualize mesh domains and layerss for debugging or analysis.
    """

    SAVE_STRING = "tlm_lx={0:.1f}_ly={1:.1f}_H={2:.2f}_lvl={3:.0f}_lyr={4:.0f}"

    def __init__(
        self,
        lx: float,
        ly: float,
        coarse_mesh_size: float,
        refinement_levels: int = 1,
        layers: int = 0,
    ):
        """
        Initialize the TwoLevelMesh with domain size, mesh size, refinement, and layers.

        Args:
            lx (float): Length of the domain in the x-direction.
            ly (float): Length of the domain in the y-direction.
            coarse_mesh_size (float): Mesh size for the coarse mesh.
            refinement_levels (int, optional): Number of uniform refinements for the fine mesh. Defaults to 1.
            layers(int, optional): Number of layers for domain decomposition. Defaults to 0.
        """
        self.lx = lx
        self.ly = ly
        self.coarse_mesh_size = coarse_mesh_size
        self.refinement_levels = refinement_levels
        self.fine_mesh, self.coarse_mesh = self.create_conforming_meshes()
        self.subdomains = self.get_subdomains()
        self.layers= layers
        for layer_idx in range(1, layers+ 1):
            self.extend_subdomains(layer_idx=layer_idx)

    def create_conforming_meshes(self) -> tuple[ngs.Mesh, ngs.Mesh]:
        """
        Create conforming coarse and fine meshes by refining the coarse mesh.
        Args:
            lx, ly: Rectangle dimensions.
            coarse_mesh_size: Mesh size for the coarse mesh.
            refinement_levels: Number of uniform refinements for the fine mesh.
        Returns:
            fine_mesh (ngs.Mesh): Refined NGSolve mesh.
            coarse_mesh (ngs.Mesh): Original coarse NGSolve mesh.
        """
        # Create rectangle geometry
        domain = OCCRectangle(self.lx, self.ly)
        face = domain.Face()

        # Set boundary labels (as before)
        face.edges.Max(occ.Y).name = BoundaryName.TOP.value
        face.edges.Min(occ.Y).name = BoundaryName.BOTTOM.value
        face.edges.Max(occ.X).name = BoundaryName.RIGHT.value
        face.edges.Min(occ.X).name = BoundaryName.LEFT.value

        geo = occ.OCCGeometry(face, dim=2)

        # Generate coarse mesh
        ngm_coarse = geo.GenerateMesh(minh=coarse_mesh_size, maxh=coarse_mesh_size)
        coarse_mesh = ngs.Mesh(ngm_coarse)

        # Make a copy for the fine mesh and refine in-place
        fine_mesh = copy.deepcopy(coarse_mesh)
        for _ in range(self.refinement_levels):
            fine_mesh.Refine()

        return fine_mesh, coarse_mesh

    def get_subdomains(self):
        """
        Identify and return the mapping of coarse mesh elements to their corresponding fine mesh domains.

        Returns:
            dict: Mapping of coarse elements to their fine mesh interior elements and edges.
        """
        subdomains = {}
        coarse_elements = self.coarse_mesh.Elements()
        interior_indices = np.arange(self.fine_mesh.ne)  # indices of fine elements
        fine_elements_vertices = np.zeros(
            (self.fine_mesh.ne, 3, 2)
        )  # el X vertices X coords
        for i, fine_el in enumerate(self.fine_mesh.Elements(ngs.VOL)):
            fine_v1 = self.fine_mesh.vertices[fine_el.vertices[0].nr].point
            fine_v2 = self.fine_mesh.vertices[fine_el.vertices[1].nr].point
            fine_v3 = self.fine_mesh.vertices[fine_el.vertices[2].nr].point
            fine_elements_vertices[i, 0, :] = fine_v1
            fine_elements_vertices[i, 1, :] = fine_v2
            fine_elements_vertices[i, 2, :] = fine_v3

        for subdomain in coarse_elements:  # coarse elements are taken to be subdomains
            interior_indices_copy = np.copy(
                interior_indices
            )  # copy of fine indices to filter
            fine_elements_vertices_copy = np.copy(fine_elements_vertices)
            coarse_v1 = self.coarse_mesh.vertices[subdomain.vertices[0].nr].point
            coarse_v2 = self.coarse_mesh.vertices[subdomain.vertices[1].nr].point
            coarse_v3 = self.coarse_mesh.vertices[subdomain.vertices[2].nr].point
            for vertex_idx in range(3):
                # look at first vertices
                fine_elements_vertex = fine_elements_vertices_copy[:, vertex_idx, :]

                # find fine elements that have this vertex within the coarse element
                mask = self._vectorized_point_in_triangle(
                    fine_elements_vertex, coarse_v1, coarse_v2, coarse_v3
                )

                # get rid of fine elements that do not have this vertex within the coarse element
                fine_elements_vertices_copy = fine_elements_vertices_copy[mask]
                interior_indices_copy = interior_indices_copy[mask]
            interior_elements = [
                self.fine_mesh[ngs.ElementId(ngs.VOL, idx)]
                for idx in interior_indices_copy
            ]
            subdomains[subdomain] = {
                "interior": interior_elements,
                "edges": self.get_subdomain_domain_edges(subdomain, interior_elements),
            }
        return subdomains

    def get_subdomain_domain_edges(self, subdomain, interior_elements):
        """
        Find all fine mesh edges that lie on the edges of a given coarse mesh element.

        Args:
            subdomain: The coarse mesh element.
            interior_elements: List of fine mesh elements inside the coarse element.

        Returns:
            list: Fine mesh edges that lie on the coarse element's edges.
        """
        subdomain_edges = set()
        for coarse_edge in subdomain.edges:
            for el in interior_elements:
                mesh_el = self.fine_mesh[el]
                fine_edges = self._get_edges_on_subdomain_edge(coarse_edge, mesh_el)
                subdomain_edges.update(fine_edges)

        return list(subdomain_edges)

    def extend_subdomains(self, layer_idx: int = 1):
        """
        Extend the subdomains by adding layers of fine mesh elements.

        Args:
            layer_idx (int, optional): The new layer's index. Defaults to 1.
        """
        for subdomain_data in self.subdomains.values():
            interior_edges = set()
            domain_elements = copy.copy(subdomain_data["interior"])
            for prev_layer_idx in range(1, layer_idx):
                domain_elements += subdomain_data[f"layer_{prev_layer_idx}"]
            for el in domain_elements:
                mesh_el = self.fine_mesh[el]
                for fine_edge in mesh_el.edges:
                    interior_edges.add(fine_edge.nr)

            layer_elements = set()
            for el in domain_elements:
                mesh_el = self.fine_mesh[el]
                vertices = mesh_el.vertices
                for v in vertices:
                    mesh_v = self.fine_mesh[v]
                    for edge in mesh_v.edges:
                        if edge.nr not in interior_edges:
                            mesh_e = self.fine_mesh[edge]
                            layer_elements.update(mesh_e.elements)
            subdomain_data[f"layer_{layer_idx}"] = list(layer_elements)

    # saving
    def save(self):
        """
        Save the mesh, metadata, and subdomain information to disk.

        Args:
            file_name (str, optional): Name of the file to save the mesh and metadata. Defaults to "".
        """
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving TwoLevelMesh to {self.save_dir}...")
        self._save_metadata()
        self._save_meshes()
        self._save_subdomains()

    def _save_metadata(self):
        """
        Save mesh and domain metadata to a JSON file.
        """
        metadata = {
            "lx": self.lx,
            "ly": self.ly,
            "coarse_mesh_size": self.coarse_mesh_size,
            "refinement_levels": self.refinement_levels,
            "layers": self.layers,
        }
        metadata_path = self.save_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def _save_meshes(self, save_vtk: bool = True):
        """
        Saves the fine and coarse mesh data to disk in both .vol and optionally .vtk formats.

        Args:
            save_vtk (bool, optional): If True, saves the meshes in VTK format as well. Defaults to True.

        Side Effects:
            - Prints information about the fine and coarse meshes (number of elements, vertices, and edges).
            - Saves the fine and coarse meshes to .vol files in the DATA_DIR directory.
            - If save_vtk is True, also saves the meshes in VTK format with the specified file name prefix.
        """
        self.fine_mesh.ngmesh.Save(str(self.save_dir / "fine_mesh.vol"))
        self.coarse_mesh.ngmesh.Save(str(self.save_dir / "coarse_mesh.vol"))
        if save_vtk:
            vtk = ngs.VTKOutput(
                self.fine_mesh,
                coefs=[],
                names=[],
                filename=str(self.save_dir / "fine_mesh"),
            )
            vtk.Do()
            vtk = ngs.VTKOutput(
                self.coarse_mesh,
                coefs=[],
                names=[],
                filename=str(self.save_dir / "coarse_mesh"),
            )
            vtk.Do()
            print("Fine mesh saved:")
            print(f"\tNumber of elements: {self.fine_mesh.ne}")
            print(f"\tNumber of vertices: {self.fine_mesh.nv}")
            print(f"\tNumber of edges: {len(self.fine_mesh.edges)}")
            print("Coarse mesh saved:")
            print(f"\tNumber of elements: {self.coarse_mesh.ne}")
            print(f"\tNumber of vertices: {self.coarse_mesh.nv}")
            print(f"\tNumber of edges: {len(self.coarse_mesh.edges)}")

    def _save_subdomains(self):
        """
        Save subdomains to a file.
        """
        subdomains_path = self.save_dir / "subdomains.json"
        with open(subdomains_path, "w") as f:
            subdomains_picklable = {
                int(subdomain.nr): {
                    "interior": [el.nr for el in subdomain_data["interior"]],
                    "edges": [edge.nr for edge in subdomain_data["edges"]],
                    **{
                        f"layer_{layer_idx}": [
                            el.nr for el in subdomain_data.get(f"layer_{layer_idx}", [])
                        ]
                        for layer_idx in range(1, self.layers+ 1)
                    },
                }
                for subdomain, subdomain_data in self.subdomains.items()
            }
            json.dump(subdomains_picklable, f, indent=4)

    # loading
    @classmethod
    def load(
        cls,
        lx: float,
        ly: float,
        coarse_mesh_size: float,
        refinement_levels: int,
        layers: int,
    ):
        """
        Load the mesh and metadata from files and return a TwoLevelMesh instance.

        Args:
            file_name (str, optional): Name of the file to load the mesh and metadata. Defaults to "".

        Returns:
            TwoLevelMesh: Loaded TwoLevelMesh instance.
        """
        folder_name = cls.SAVE_STRING.format(
            lx, ly, coarse_mesh_size, refinement_levels, layers
        )
        fp = DATA_DIR / folder_name
        if fp.exists():
            print(f"Loading TwoLevelMesh from {fp}...")
            obj = cls.__new__(cls)
            obj._load_metadata(fp)
            obj._load_meshes(fp)
            obj._load_subdomains(fp)
        else:
            raise FileNotFoundError(f"Metadata file {fp} does not exist.")
        return obj

    def _load_metadata(self, fp: Path):
        """
        Load mesh and domain metadata from a JSON file.
        """
        metadata_path = fp / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file {metadata_path} does not exist.")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        self.lx = metadata["lx"]
        self.ly = metadata["ly"]
        self.coarse_mesh_size = metadata["coarse_mesh_size"]
        self.refinement_levels = metadata["refinement_levels"]
        self.layers= metadata["layers"]

    def _load_meshes(self, fp: Path):
        """
        Load fine and coarse meshes from disk.
        """
        fine_mesh_path = fp / "fine_mesh.vol"
        coarse_mesh_path = fp / "coarse_mesh.vol"
        if not fine_mesh_path.exists() or not coarse_mesh_path.exists():
            raise FileNotFoundError(
                f"Mesh files {fine_mesh_path} or {coarse_mesh_path} do not exist."
            )
        self.fine_mesh = ngs.Mesh(str(fine_mesh_path))
        self.coarse_mesh = ngs.Mesh(str(coarse_mesh_path))
        print("Fine mesh loaded.")
        print(f"\tNumber of elements: {self.fine_mesh.ne}")
        print(f"\tNumber of vertices: {self.fine_mesh.nv}")
        print(f"\tNumber of edges: {len(self.fine_mesh.edges)}")
        print("Coarse mesh loaded.")
        print(f"\tNumber of elements: {self.coarse_mesh.ne}")
        print(f"\tNumber of vertices: {self.coarse_mesh.nv}")
        print(f"\tNumber of edges: {len(self.coarse_mesh.edges)}")

    def _load_subdomains(self, fp: Path):
        """
        Load subdomains from a file and reconstruct the mapping.

        Args:
            file_name (str, optional): Name of the file to load the subdomains. Defaults to "".
        """
        subdomains_path = fp / "subdomains.json"
        if not subdomains_path.exists():
            raise FileNotFoundError(
                f"subdomains file {subdomains_path} does not exist."
            )
        with open(subdomains_path, "r") as f:
            self.subdomains = json.load(f)

        # Convert keys back to elements
        self.subdomains = {
            self.coarse_mesh[ngs.ElementId(ngs.VOL, int(subdomain))]: {
                "interior": [
                    self.fine_mesh[ngs.ElementId(ngs.VOL, el)]
                    for el in subdomain_data["interior"]
                ],
                "edges": [
                    self.fine_mesh.edges[edge] for edge in subdomain_data["edges"]
                ],
                **{
                    f"layer_{layer_idx}": [
                        self.fine_mesh[ngs.ElementId(ngs.VOL, el)]
                        for el in subdomain_data.get(f"layer_{layer_idx}", [])
                    ]
                    for layer_idx in range(1, self.layers+ 1)
                },
            }
            for subdomain, subdomain_data in self.subdomains.items()
        }

    @property
    def save_dir(self):
        """
        Directory where the mesh and subdomain data are saved.
        """
        folder_name = self.SAVE_STRING.format(
            self.lx,
            self.ly,
            self.coarse_mesh_size,
            self.refinement_levels,
            self.layers,
        )
        return DATA_DIR / folder_name

    @save_dir.setter
    def save_dir(self, folder_name: str):
        """
        Set the folder where the mesh and subdomain data are saved.

        Args:
            folder_name (str): Name of the folder.
        """
        self._save_folder = DATA_DIR / folder_name
        if not self._save_folder.exists():
            self._save_folder.mkdir(parents=True, exist_ok=True)


    # plotting
    def plot_domains(
        self,
        domains: list | int = 1,
        plot_layers: bool = False,
        opacity: float = 0.9,
        fade_factor: float = 1.5,
    ):
        """
        Visualize the subdomains and optional layers using matplotlib.

        Args:
            domains (list or int, optional): Which domains to plot. Defaults to 1.
            plot_layers (bool, optional): Whether to plot layers. Defaults to False.
            opacity (float, optional): Opacity of the domain fill. Defaults to 0.9.
            fade_factor (float, optional): Controls fading of layers. Defaults to 1.5.

        Returns:
            tuple: (figure, ax) Matplotlib figure and axis.
        """
        domains_int_toggle = isinstance(domains, int)
        domains_list_toggle = isinstance(domains, list)
        figure, ax = plt.subplots(figsize=(8, 6))
        for i, (subdomain, subdomain_data) in enumerate(self.subdomains.items()):
            domain_elements = subdomain_data["interior"]
            if domains_int_toggle:
                if i % domains != 0:
                    continue
            elif domains_list_toggle:
                if subdomain not in domains:
                    continue
            fillcolor = self.subdomain_colors[i % len(self.subdomain_colors)]
            for domain_el in domain_elements:
                self.plot_element(
                    ax,
                    domain_el,
                    self.fine_mesh,
                    fillcolor=fillcolor,
                    alpha=opacity,
                    edgecolor="black",
                    label="Coarse Element",
                )
            if plot_layers:
                for layer_idx in range(1, self.layers+ 1):
                    layer_elements = subdomain_data.get(f"layer_{layer_idx}", [])
                    alpha_value = opacity / (
                        1 + (layer_idx / self.layers) ** fade_factor
                    )
                    for layer_el in layer_elements:
                        self.plot_element(
                            ax,
                            layer_el,
                            self.fine_mesh,
                            fillcolor=fillcolor,
                            alpha=alpha_value,
                            edgecolor="black",
                            label=f"layersLayer {layer_idx} Element",
                        )
        return figure, ax

    @staticmethod
    def plot_element(
        ax: Axes,
        element,
        mesh,
        fillcolor="blue",
        edgecolor="black",
        alpha=1.0,
        label=None,
    ):
        """
        Plot a single mesh element on a matplotlib axis.

        Args:
            ax (Axes): Matplotlib axis to plot on.
            element: ElementId of the element to plot.
            mesh: NGSolve mesh containing the element.
            fillcolor (str, optional): Color for the element. Defaults to "blue".
            edgecolor (str, optional): Color for the element edges. Defaults to "black".
            alpha (float, optional): Opacity of the fill. Defaults to 1.0.
            label (str, optional): Label for the element (optional).
        """
        vertices = [mesh[v].point for v in mesh[element].vertices]
        polygon = Polygon(
            vertices,
            closed=True,
            fill=True if fillcolor else False,
            facecolor=fillcolor,
            edgecolor=edgecolor,
            label=label,
            alpha=alpha,
        )
        ax.add_patch(polygon)

    @property
    def subdomain_colors(self):
        """
        List of colors for plotting domains.
        """
        if not hasattr(self, "_subdomain_colors"):
            self._subdomain_colors = CUSTOM_COLORS_SIMPLE
        return self._subdomain_colors

    @subdomain_colors.setter
    def subdomain_colors(self, colors):
        """
        Set custom colors for the domains.

        Args:
            colors (list): List of color strings.
        """
        if not isinstance(colors, list):
            raise ValueError("Domain colors must be a list.")
        self._subdomain_colors = colors

    # supporting methods
    def _get_edges_on_subdomain_edge(self, coarse_edge, mesh_element):
        """
        Find all fine edges that lie on a given subdomain (coarse mesh) edge.

        Args:
            coarse_edge: The coarse mesh edge.
            mesh_element: The fine mesh element.

        Returns:
            set: Fine edges that lie on the coarse edge.
        """
        fine_edges = set()
        vc_1, vc_2 = self.coarse_mesh.edges[coarse_edge.nr].vertices
        pc_1 = self.coarse_mesh.vertices[vc_1.nr].point
        pc_2 = self.coarse_mesh.vertices[vc_2.nr].point
        edge_dir = np.array(pc_2) - np.array(pc_1)
        edge_dir_3d = np.array((edge_dir[0], edge_dir[1], 0))  # Convert to 3D vector
        edge_length = np.linalg.norm(edge_dir)
        edge_dir /= edge_length  # Normalize the direction vector
        for fine_edge in mesh_element.edges:
            vf_1, vf_2 = self.fine_mesh.edges[fine_edge.nr].vertices
            pf_1 = self.fine_mesh.vertices[vf_1.nr].point
            pf_2 = self.fine_mesh.vertices[vf_2.nr].point
            fine_edge_dir = np.array(pf_2) - np.array(pf_1)
            fine_edge_length = np.linalg.norm(fine_edge_dir)
            fine_edge_dir /= fine_edge_length
            # Check if the fine edge direction is parallel to the coarse edge direction
            if np.allclose(fine_edge_dir, edge_dir, atol=1e-10):
                # Check if the fine edge lies on the coarse edge
                d_vec = np.array(pf_1 + (0,)) - np.array(pc_1 + (0,))
                cross_vec = np.cross(edge_dir_3d, d_vec)
                t = np.linalg.norm(cross_vec)  # Distance from coarse edge to fine edge
                if np.isclose(t, 0, atol=1e-10):
                    fine_edges.add(fine_edge)
        return fine_edges

    @staticmethod
    def _vectorized_point_in_triangle(points, a, b, c):
        """
        Vectorized check if points are inside the triangle defined by (a, b, c).

        Args:
            points (np.ndarray): Array of points to check (N, 2).
            a, b, c (array-like): Triangle vertices.

        Returns:
            np.ndarray: Boolean mask of points inside the triangle.
        """
        # Ensure points is (N, 2)
        points = np.asarray(points)

        N = points.shape[0]

        A = TwoLevelMesh._vectorized_area(
            np.tile(a, (N, 1)), np.tile(b, (N, 1)), np.tile(c, (N, 1))
        )
        A1 = TwoLevelMesh._vectorized_area(
            points, np.tile(b, (N, 1)), np.tile(c, (N, 1))
        )
        A2 = TwoLevelMesh._vectorized_area(
            np.tile(a, (N, 1)), points, np.tile(c, (N, 1))
        )
        A3 = TwoLevelMesh._vectorized_area(
            np.tile(a, (N, 1)), np.tile(b, (N, 1)), points
        )

        return np.abs(A - (A1 + A2 + A3)) < 1e-9

    @staticmethod
    def _vectorized_area(p1, p2, p3):
        """
        Compute the area of triangles defined by points p1, p2, p3.

        Args:
            p1, p2, p3 (np.ndarray): Arrays of triangle vertices (N, 2).

        Returns:
            np.ndarray: Areas of the triangles.
        """
        return 0.5 * np.abs(
            (
                p1[:, 0] * (p2[:, 1] - p3[:, 1])
                + p2[:, 0] * (p3[:, 1] - p1[:, 1])
                + p3[:, 0] * (p1[:, 1] - p2[:, 1])
            )
        )


# Example usage:
if __name__ == "__main__":
    lx, ly = 1.0, 1.0
    coarse_mesh_size = 0.15
    refinement_levels = 2
    layers= 2
    two_mesh = TwoLevelMesh(
        lx, ly, coarse_mesh_size, refinement_levels=refinement_levels, layers=layers
    )
    two_mesh.save()  # Save the mesh and subdomains
    # two_mesh = TwoLevelMesh.load(
    #     lx, ly, coarse_mesh_size, refinement_levels, layers
    # )
    figure, ax = two_mesh.plot_domains(domains=7, plot_layers=True)
    plt.show()
