import copy
import json
from enum import Enum
from itertools import cycle
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
    A class for generating, managing, and visualizing two-level conforming meshes (coarse and fine) for rectangular domains, with support for layering domain decomposition and mesh serialization.
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
    ZORDERS = {
        "elements": 1.0,
        "layers": 1.5,
        "edges": 2.0,
        "vertices": 3.0,
    }

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
        self.connected_components = self.get_connected_components()
        self.connected_component_tree = self.get_connected_component_tree()
        self.layers = layers
        for layer_idx in range(1, layers + 1):
            self.extend_subdomains(layer_idx)
        print(self)

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
        ngm_coarse = geo.GenerateMesh(
            minh=self.coarse_mesh_size, maxh=self.coarse_mesh_size
        )
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

    def get_subdomain_domain_edges(self, subdomain, interior_elements) -> dict:
        """
        Find all fine mesh edges that lie on the edges of a given coarse mesh element.

        Args:
            subdomain: The coarse mesh element.
            interior_elements: List of fine mesh elements inside the coarse element.

        Returns:
            dict: Fine mesh edges that lie on the coarse element's edges.
        """
        subdomain_edges = {}
        for coarse_edge in subdomain.edges:
            subdomain_edges[coarse_edge] = []
            for el in interior_elements:
                mesh_el = self.fine_mesh[el]
                fine_edges = self._get_edges_on_subdomain_edge(coarse_edge, mesh_el)
                subdomain_edges[coarse_edge] += fine_edges
        return subdomain_edges

    def get_connected_component_tree(self) -> dict:
        """
        Finds the tree of connected components in the fine mesh based on the coarse mesh subdomains.

        Note a 2D mesh does not have faces so the fine edge lists will be empty.
        Returns:
            dict: A dictionary with the following structure:
                {
                    coarse_node_i1: {
                        coarse_edge_j1:{
                            "fine_edges": [fine_edge_1, fine_edge_2, ...],
                            "fine_vertices": [fine_vertex_1, fine_vertex_2, ...],
                            coarse_face_k1: {
                                "fine_faces": [fine_face_1, fine_face_2, ...],
                                "fine_edges": [fine_edge_1, fine_edge_2, ...],
                                "fine_vertices": [fine_vertex_1, fine_vertex_2, ...]
                            },
                        },
                        coarse_edge_j2:{...},
                        ...,
                        coarse_edge_jm:{...},
                    }
                    coarse_node_i2: {...},
                    ...,
                    coarse_node_in: {...}
                }
        """
        component_tree = {}
        all_coarse_nodes = set(self.coarse_mesh.vertices)
        for coarse_node in all_coarse_nodes:
            component_tree[coarse_node] = {}
        
        for subdomain, subdomain_data in self.subdomains.items():
            for coarse_node in subdomain.vertices:
                for coarse_edge in self.coarse_mesh[coarse_node].edges:
                    if (coarse_edge_d := component_tree[coarse_node].get(coarse_edge, None)) is None:
                        component_tree[coarse_node][coarse_edge] = {}
                        coarse_edge_d = component_tree[coarse_node][coarse_edge] 
                    if (fine_edges := subdomain_data["edges"].get(coarse_edge, None)) is not None:
                        coarse_edge_d["fine_edges"] = fine_edges
                        edge_vertices = set()
                        for fine_edge in fine_edges:
                            for vertex in self.fine_mesh[fine_edge].vertices:
                                if vertex not in all_coarse_nodes:
                                    edge_vertices.add(vertex)
                        coarse_edge_d["fine_vertices"] = list(edge_vertices)
            
        return component_tree

    def get_connected_components(self) -> dict:
        """
        Finds all the connected components in the fine mesh based on the coarse mesh subdomains.

        That is,
        1) each coarse node,
        2) for each subdomain edge all fine edges and fine vertices that belong to it (except for those vertices belonging to 1)
        3) for each subdomain face all fine faces, fine edges, fine vertices that belong to it (except for those belonging to 1 & 2)

        Returns:
            dict: A dictionary mapping each coarse mesh element to its fine mesh elements.
        """
        connected_components = {}

        connected_components["coarse_nodes"] = []
        coarse_nodes_processed = set()

        connected_components["edges"] = []
        subdomain_edges_processed = set()

        # NOTE: this code is commented out until 3D meshes are implemented
        connected_components["faces"] = []
        subdomain_faces_processed = set()

        # figure, ax = plt.subplots()
        for subdomain, subdomain_data in self.subdomains.items():
            # coarse nodes
            subdomain_coarse_nodes = subdomain.vertices
            for subdomain_coarse_node in subdomain_coarse_nodes:
                if subdomain_coarse_node not in coarse_nodes_processed:
                    connected_components["coarse_nodes"].append(subdomain_coarse_node)
                    coarse_nodes_processed.add(subdomain_coarse_node)

            # subdomain edges
            for subdomain_edge, fine_edges in subdomain_data["edges"].items():
                # add fine edges
                subdomain_edge_components = set(fine_edges)
                for fine_edge in fine_edges:
                    # add fine vertices
                    for vertex in self.fine_mesh[fine_edge].vertices:
                        # coarse nodes are not added to edges
                        if vertex not in coarse_nodes_processed:
                            subdomain_edge_components.add(vertex)

                if subdomain_edge not in subdomain_edges_processed:
                    connected_components["edges"].append(
                        list(subdomain_edge_components)
                    )
                    subdomain_edges_processed.add(subdomain_edge)

                # NOTE: this code is commented out until 3D meshes are implemented
                # # faces
                # if (subdomain_faces := getattr(subdomain, "faces")) is not None:
                #     for subdomain_face, fine_faces in subdomain_faces.items():
                #         subdomain_face_components = set()
                #         subdomain_faces.remove(subdomain_face)
                #         if len(subdomain_faces) == 0:
                #             break
                #         for fine_face in fine_faces:
                #             # add face
                #             subdomain_face_components.add(fine_face)

                #             # add edges
                #             for edge in fine_face.edges:
                #                 if edge not in subdomain_edge_components:
                #                     subdomain_face_components.add(edge)

                #             # add vertices
                #             for vertex in fine_face.vertices:
                #                 if vertex not in coarse_nodes:
                #                     subdomain_face_components.add(vertex)

        return connected_components

    def extend_subdomains(self, layer_idx):
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

    # in-plca mesh refinement
    def refine_coarse_mesh(self):
        """
        Refine the coarse mesh in-place to create a fine mesh that can be used for the fespace construction 
        and obtaining the prolongation operator
        """
        for _ in range(self.refinement_levels):
            self.coarse_mesh.Refine()
    
    # meta info string
    def __str__(self):
        return (
            f"Fine mesh:"
            f"\n\telements: {self.fine_mesh.ne}"
            f"\n\tvertices: {self.fine_mesh.nv}"
            f"\n\tedges: {len(self.fine_mesh.edges)}"
            f"\nCoarse mesh:"
            f"\n\telements: {self.coarse_mesh.ne}"
            f"\n\tvertices: {self.coarse_mesh.nv}"
            f"\n\tedges: {len(self.coarse_mesh.edges)}"
        )

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
        # self._save_subdomains()

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
                    "edges": {
                        int(coarse_edge.nr): [
                            edge.nr for edge in subdomain_data["edges"][coarse_edge]
                        ]
                        for coarse_edge in subdomain_data["edges"]
                    },
                    **{
                        f"layer_{layer_idx}": [
                            el.nr for el in subdomain_data.get(f"layer_{layer_idx}", [])
                        ]
                        for layer_idx in range(1, self.layers + 1)
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
            print(f"\tloaded metadata")
            obj._load_meshes(fp)
            print(f"\tloaded meshes")
            # obj._load_subdomains(fp) # TODO: fix this
            setattr(obj, "subdomains", obj.get_subdomains())
            print(f"\tcalculated subdomains")
            for layer_idx in range(1, obj.layers + 1):
                obj.extend_subdomains(layer_idx)
            print(f"\textended subdomains with {obj.layers} layers")
            setattr(obj, "connected_components", obj.get_connected_components())
            print(f"\tcalculated connected components")
            setattr(obj, "connected_component_tree", obj.get_connected_component_tree())
            print(f"\tcalculated connected component tree")
            print("Finished loading TwoLevelMesh.")
            print(obj)
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
        self.layers = metadata["layers"]

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

    def _load_subdomains(self, fp: Path):
        """
        Load subdomains from a file and reconstruct the mapping.

        Args:
            file_name (str, optional): Name of the file to load the subdomains. Defaults to "".
        """
        # TODO: FIX SUBDOMAINS LOADING
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
                "edges": {
                    self.coarse_mesh.edges[int(coarse_edge_nr)]: [
                        self.fine_mesh.edges[edge_nr]
                        for edge_nr in subdomain_data["edges"][coarse_edge_nr]
                    ]
                    for coarse_edge_nr in subdomain_data["edges"]
                },
                **{
                    f"layer_{layer_idx}": [
                        self.fine_mesh[ngs.ElementId(ngs.VOL, el)]
                        for el in subdomain_data.get(f"layer_{layer_idx}", [])
                    ]
                    for layer_idx in range(1, self.layers + 1)
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
    def plot_mesh(self, ax: Axes, mesh_type: str = "fine"):
        """
        Plot the fine or coarse mesh using matplotlib.

        Args:
            mesh_type (str, optional): Type of mesh to plot ('fine' or 'coarse'). Defaults to 'fine'.

        Returns:
            tuple: (figure, ax) Matplotlib figure and axis.
        """
        if mesh_type == "fine":
            mesh = self.fine_mesh
            linewidth = 1.0
            alpha = 1.0
            fillcolor = "lightgray"
            edgecolor = "black"
        elif mesh_type == "coarse":
            mesh = self.coarse_mesh
            linewidth = 2.0
            alpha = 0.5
            fillcolor = "lightblue"
            edgecolor = "darkblue"
        else:
            raise ValueError("mesh_type must be 'fine' or 'coarse'.")

        for el in mesh.Elements():
            self.plot_element(
                ax,
                el,
                mesh,
                fillcolor=fillcolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
            )
        return ax

    def plot_domains(
        self,
        ax: Axes,
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
        for i, (subdomain, subdomain_data) in enumerate(self.subdomains.items()):
            #plot coarse mesh element without fill and thick border
            self.plot_element(
                ax,
                subdomain,
                self.coarse_mesh,
                fillcolor=None,
                edgecolor="black",
                alpha=1.0,
                linewidth=2.0,
                zorder=TwoLevelMesh.ZORDERS["edges"] + 0.1,
            )
            domain_elements = subdomain_data["interior"]
            if domains_int_toggle:
                if i % domains != 0:
                    continue
            elif domains_list_toggle:
                if subdomain.nr not in domains:
                    continue
            fillcolor = next(self.subdomain_colors)
            for domain_el in domain_elements:
                self.plot_element(
                    ax,
                    domain_el,
                    self.fine_mesh,
                    fillcolor=fillcolor,
                    alpha=opacity,
                    edgecolor="black",
                )
            if plot_layers:
                for layer_idx in range(1, self.layers + 1):
                    layer_elements = subdomain_data.get(f"layer_{layer_idx}", [])
                    alpha_value = opacity / (
                        1 + (layer_idx / self.layers) ** fade_factor
                    ) / 4
                    for layer_el in layer_elements:
                        self.plot_element(
                            ax,
                            layer_el,
                            self.fine_mesh,
                            fillcolor="black",
                            alpha=alpha_value,
                            edgecolor="black",
                            label=f"layersLayer {layer_idx} Element",
                            zorder=TwoLevelMesh.ZORDERS["layers"],
                        )
        return ax

    def plot_connected_components(self, ax: Axes):
        """
        Plot the connected components of the fine mesh based on the coarse mesh subdomains.

        Args:
            ax (Axes): Matplotlib axis to plot on.

        Returns:
            Axes: The matplotlib axis with the plotted connected components.
        """
        # coarse nodes (correspond to fine vertices)

        self.plot_vertices(
            ax,
            self.connected_components["coarse_nodes"],
            self.fine_mesh,
            color="red",
            marker="o",
            markersize=20,
        )

        # edges
        for components in self.connected_components["edges"]:
            for component in components:
                mesh_comp = self.fine_mesh[component]
                try:  # edge component
                    _ = mesh_comp.vertices
                    self.plot_edges(
                        ax,
                        [mesh_comp],
                        self.fine_mesh,
                        color="blue",
                        linewidth=1.5,
                        linestyle="-",
                    )
                except TypeError:  # vertex component
                    self.plot_vertices(
                        ax,
                        [component],
                        self.fine_mesh,
                        color="green",
                        marker="x",
                        markersize=15,
                    )

        # faces
        for face_component in self.connected_components["faces"]:
            for face in face_component:
                raise NotImplementedError(
                    "Plotting connected components for faces is not implemented yet."
                )
        return ax
    
    def plot_connected_component_tree(self, ax: Axes):
        """
        Plot the connected component tree of the fine mesh based on the coarse mesh subdomains.

        Args:
            ax (Axes): Matplotlib axis to plot on.

        Returns:
            Axes: The matplotlib axis with the plotted connected component tree.
        """
        plotted_edges = set()
        for coarse_node, edges in self.connected_component_tree.items():
            color = next(self.subdomain_colors)
            # plot big coarse node
            self.plot_vertices(
                ax,
                [coarse_node],
                self.coarse_mesh,
                color=color,
                marker="x",
                markersize=50,
            )
            for coarse_edge, edge_data in edges.items():
                # plot thick coarse edge showing the connected component
                self.plot_edges(
                    ax,
                    [coarse_edge],
                    self.coarse_mesh,
                    color=color,
                    alpha=0.25,
                    linewidth=8.0,
                    zorder=TwoLevelMesh.ZORDERS["edges"] - 0.1,
                    linestyle="-" if coarse_edge not in plotted_edges else "--",
                )
                plotted_edges.add(coarse_edge)

                # plot fine edges and vertices
                self.plot_edges(
                    ax,
                    edge_data["fine_edges"],
                    self.fine_mesh,
                    color="black",
                    linewidth=0.5,
                    linestyle="-",
                )
                self.plot_vertices(
                    ax,
                    edge_data["fine_vertices"],
                    self.fine_mesh,
                    color="green",
                    marker="x",
                    markersize=15,
                )
        return ax

    @staticmethod
    def plot_element(
        ax: Axes,
        element,
        mesh,
        **kwargs,
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
            fill=True if kwargs.get("fillcolor") else False,
            facecolor=kwargs.get("fillcolor", "blue"),
            edgecolor=kwargs.get("edgecolor", "black"),
            label=kwargs.get("label"),
            alpha=kwargs.get("alpha", 1.0),
            linewidth=kwargs.get("linewidth", 1.5),
            zorder=kwargs.get("zorder", TwoLevelMesh.ZORDERS["elements"]),
        )
        ax.add_patch(polygon)

    @staticmethod
    def plot_edges(
        ax: Axes,
        edges: list,
        mesh,
        **kwargs,
    ):
        """
        Plot a single mesh edge on a matplotlib axis.

        Args:
            ax (Axes): Matplotlib axis to plot on.
            edge: EdgeId of the edge to plot.
            mesh: NGSolve mesh containing the edge.
            color (str, optional): Color for the edge. Defaults to "black".
            linewidth (float, optional): Width of the edge line. Defaults to 1.5.
            linestyle (str, optional): Style of the edge line. Defaults to "-".
            label (str, optional): Label for the edge (optional).
        """
        for edge in edges:
            vertices = [mesh[v].point for v in mesh[edge].vertices]
            ax.plot(
                [v[0] for v in vertices],
                [v[1] for v in vertices],
                color=kwargs.get("color", "black"),
                linewidth=kwargs.get("linewidth", 1.5),
                linestyle=kwargs.get("linestyle", "-"),
                label=kwargs.get("label"),
                zorder=kwargs.get("zorder", TwoLevelMesh.ZORDERS["edges"]),
            )

    @staticmethod
    def plot_vertices(
        ax: Axes,
        vertices: list,
        mesh,
        **kwargs,
    ):
        """
        Plot a single mesh node on a matplotlib axis.

        Args:
            ax (Axes): Matplotlib axis to plot on.
            node: NodeId of the node to plot.
            mesh: NGSolve mesh containing the node.
            color (str, optional): Color for the node. Defaults to "red".
            marker (str, optional): Marker style for the node. Defaults to "o".
            markersize (int, optional): Size of the marker. Defaults to 5.
            label (str, optional): Label for the node (optional).
        """
        coarse_node_coords = np.array([mesh.vertices[v.nr].point for v in vertices])
        ax.scatter(
            coarse_node_coords[:, 0],
            coarse_node_coords[:, 1],
            color=kwargs.get("color", "red"),
            marker=kwargs.get("marker", "o"),
            s=kwargs.get("markersize", 5),
            label=kwargs.get("label"),
            zorder=kwargs.get("zorder", TwoLevelMesh.ZORDERS["vertices"]),
        )

    @property
    def subdomain_colors(self) -> cycle:
        """
        List of colors for plotting domains.
        """
        if not hasattr(self, "_subdomain_colors"):
            self._subdomain_colors = cycle(CUSTOM_COLORS_SIMPLE)
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
        self._subdomain_colors = cycle(colors)

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
        fine_edges = []
        vc_1, vc_2 = self.coarse_mesh.edges[coarse_edge.nr].vertices
        pc_1 = self.coarse_mesh.vertices[vc_1.nr].point
        pc_2 = self.coarse_mesh.vertices[vc_2.nr].point
        for fine_edge in mesh_element.edges:
            vf_1, vf_2 = self.fine_mesh.edges[fine_edge.nr].vertices
            pf_1 = self.fine_mesh.vertices[vf_1.nr].point
            pf_2 = self.fine_mesh.vertices[vf_2.nr].point
            p1_on_line = self._check_if_point_on_line(pf_1, pc_1, pc_2)
            p2_on_line = self._check_if_point_on_line(pf_2, pc_1, pc_2)
            if p1_on_line and p2_on_line:
                fine_edges.append(fine_edge)
        return fine_edges

    @staticmethod
    def _check_if_point_on_line(point, line_start, line_end) -> np.bool:
        """
        Check if a point is on a line segment defined by (line_start, line_end).

        Args:
            point (array-like): The point to check.
            line_start (array-like): The start point of the line segment.
            line_end (array-like): The end point of the line segment.

        Returns:
            bool: True if the point is on the line segment, False otherwise.
        """
        d1 = np.linalg.norm(np.array(point) - np.array(line_start))
        d2 = np.linalg.norm(np.array(point) - np.array(line_end))
        line_length = np.linalg.norm(np.array(line_end) - np.array(line_start))
        return np.isclose(d1 + d2, line_length, atol=1e-9)

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
    layers = 2
    # two_mesh = TwoLevelMesh(
    #     lx, ly, coarse_mesh_size, refinement_levels=refinement_levels, layers=layers
    # )
    # two_mesh.save()  # Save the mesh and subdomains
    two_mesh = TwoLevelMesh.load(lx, ly, coarse_mesh_size, refinement_levels, layers)

    # plot the meshes
    figure, ax = plt.subplots(figsize=(8, 6))
    two_mesh.plot_mesh(ax, mesh_type="fine")
    two_mesh.plot_mesh(ax, mesh_type="coarse")

    # plot the domains
    figure, ax = plt.subplots(figsize=(8, 6))
    two_mesh.plot_domains(ax, domains=1, plot_layers=True)

    # get all connected components
    figure, ax = plt.subplots(figsize=(8, 6))
    two_mesh.plot_connected_components(ax)

    # plot the connected component tree
    figure, ax = plt.subplots(figsize=(8, 6))
    two_mesh.plot_connected_component_tree(ax)
    plt.show()