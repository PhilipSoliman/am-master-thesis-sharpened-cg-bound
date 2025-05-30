import copy
import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# import netgen.occ as occ
import netgen.libngpy._NgOCC as occ
import ngsolve as ngs
import numpy as np
from matplotlib.patches import Polygon
from ngsolve.webgui import Draw

from lib.utils import (
    CUSTOM_COLORS_SIMPLE,
    get_cli_args,
    get_root,
    mpl_graph_plot_style,
    save_latex_figure,
    set_mpl_cycler,
    set_mpl_style,
)

domain_colors = CUSTOM_COLORS_SIMPLE

# create/obtain directories
DATA_DIR = get_root() / "data"
BOUNDARY_NAMES = [
    "lid",
    "floor",
    "right",
    "left",
]


def Rectangle(l, w):
    return occ.WorkPlane().Rectangle(l, w)


def create_conforming_meshes(lx, ly, coarse_mesh_size, refinement_levels=1):
    """
    Create conforming coarse and fine meshes by refining the coarse mesh.
    Args:
        lx, ly: Rectangle dimensions.
        coarse_mesh_size: Mesh size for the coarse mesh.
        refinement_levels: Number of uniform refinements for the fine mesh.
    Returns:
        fine_mesh: Refined NGSolve mesh.
        coarse_mesh: Original coarse NGSolve mesh.
    """
    # Create rectangle geometry
    domain = Rectangle(lx, ly)
    face = domain.Face()

    # Set boundary labels (as before)
    face.edges.Max(occ.Y).name = BOUNDARY_NAMES[0]
    face.edges.Min(occ.Y).name = BOUNDARY_NAMES[1]
    face.edges.Max(occ.X).name = BOUNDARY_NAMES[2]
    face.edges.Min(occ.X).name = BOUNDARY_NAMES[3]

    geo = occ.OCCGeometry(face, dim=2)

    # Generate coarse mesh
    ngm_coarse = geo.GenerateMesh(minh=coarse_mesh_size, maxh=coarse_mesh_size)
    coarse_mesh = ngs.Mesh(ngm_coarse)

    # Make a copy for the fine mesh and refine in-place
    fine_mesh = copy.deepcopy(coarse_mesh)
    for _ in range(refinement_levels):
        fine_mesh.Refine()

    return fine_mesh, coarse_mesh


def find_fine_edges_in_coarse_edge(coarse_mesh, coarse_edge, fine_mesh, mesh_element):
    """
    Find all fine edges that lie on coarse mesh edges.
    Returns:
        fine_edges: set of fine edges that lie on coarse edges.
    """
    fine_edges = set()
    vc_1, vc_2 = coarse_mesh.edges[coarse_edge.nr].vertices
    pc_1 = coarse_mesh.vertices[vc_1.nr].point
    pc_2 = coarse_mesh.vertices[vc_2.nr].point
    edge_dir = np.array(pc_2) - np.array(pc_1)
    edge_dir_3d = np.array((edge_dir[0], edge_dir[1], 0))  # Convert to 3D vector
    edge_length = np.linalg.norm(edge_dir)
    edge_dir /= edge_length  # Normalize the direction vector
    for fine_edge in mesh_element.edges:
        vf_1, vf_2 = fine_mesh.edges[fine_edge.nr].vertices
        pf_1 = fine_mesh.vertices[vf_1.nr].point
        pf_2 = fine_mesh.vertices[vf_2.nr].point
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


def find_fine_elements_within_coarse_elements(coarse_mesh, fine_mesh):
    coarse_domain_dict = {}
    coarse_elements = coarse_mesh.Elements()
    fine_indices = np.arange(fine_mesh.ne)  # indices of fine elements
    fine_elements_vertices = np.zeros((fine_mesh.ne, 3, 2))  # el X vertices X coords
    for i, fine_el in enumerate(fine_mesh.Elements(ngs.VOL)):
        fine_v1 = fine_mesh.vertices[fine_el.vertices[0].nr].point
        fine_v2 = fine_mesh.vertices[fine_el.vertices[1].nr].point
        fine_v3 = fine_mesh.vertices[fine_el.vertices[2].nr].point
        fine_elements_vertices[i, 0, :] = fine_v1
        fine_elements_vertices[i, 1, :] = fine_v2
        fine_elements_vertices[i, 2, :] = fine_v3

    for coarse_el in coarse_elements:
        fine_indices_copy = np.copy(fine_indices)  # copy of fine indices to filter
        fine_elements_vertices_copy = np.copy(fine_elements_vertices)
        coarse_v1 = coarse_mesh.vertices[coarse_el.vertices[0].nr].point
        coarse_v2 = coarse_mesh.vertices[coarse_el.vertices[1].nr].point
        coarse_v3 = coarse_mesh.vertices[coarse_el.vertices[2].nr].point
        for vertex_idx in range(3):
            # look at first vertices
            fine_elements_vertex = fine_elements_vertices_copy[:, vertex_idx, :]

            # find fine elements that have this vertex within the coarse element
            mask = vectorized_point_in_triangle(
                fine_elements_vertex, coarse_v1, coarse_v2, coarse_v3
            )

            # get rid of fine elements that do not have this vertex within the coarse element
            fine_elements_vertices_copy = fine_elements_vertices_copy[mask]
            fine_indices_copy = fine_indices_copy[mask]
        coarse_domain_dict[coarse_el] = fine_indices_copy

    return coarse_domain_dict


def vectorized_point_in_triangle(points, a, b, c):
    # Ensure points is (N, 2)
    points = np.asarray(points)

    N = points.shape[0]

    A = area(np.tile(a, (N, 1)), np.tile(b, (N, 1)), np.tile(c, (N, 1)))
    A1 = area(points, np.tile(b, (N, 1)), np.tile(c, (N, 1)))
    A2 = area(np.tile(a, (N, 1)), points, np.tile(c, (N, 1)))
    A3 = area(np.tile(a, (N, 1)), np.tile(b, (N, 1)), points)

    return np.abs(A - (A1 + A2 + A3)) < 1e-9


def area(p1, p2, p3):
    return 0.5 * np.abs(
        (
            p1[:, 0] * (p2[:, 1] - p3[:, 1])
            + p2[:, 0] * (p3[:, 1] - p1[:, 1])
            + p3[:, 0] * (p1[:, 1] - p2[:, 1])
        )
    )


def plot_element(ax, element, mesh, fillcolor="blue", edgecolor="black", alpha=1.0, label=None):
    """
    Plot a single element from the mesh.
    Args:
        ax: Matplotlib axis to plot on.
        element: ElementId of the element to plot.
        mesh: NGSolve mesh containing the element.
        fillcolor: Color for the element.
        edgecolor: Color for the element edges.
        label: Label for the element (optional).
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


def extend_domain(mesh, domain_elements):
    interior_edges = set()
    for el in domain_elements:
        mesh_el = mesh[el]
        for fine_edge in mesh_el.edges:
            interior_edges.add(fine_edge.nr)

    neighbor_elements = set()
    for el in domain_elements:
        mesh_el = mesh[el]
        vertices = mesh_el.vertices
        for v in vertices:
            mesh_v = mesh[v]
            for edge in mesh_v.edges:
                if edge.nr not in interior_edges:
                    mesh_e = mesh[edge]
                    neighbor_elements.update(mesh_e.elements)
    return list(neighbor_elements)


def get_coarse_element_edge_dofs(fine_mesh, coarse_el):
    coerse_el_edge_dofs = set()
    for coarse_edge in coarse_el.edges:
        coerse_el_edge_dofs = set()
        for fine_index in fine_indices:
            mesh_el = fine_mesh[ngs.ElementId(ngs.VOL, fine_index)]
            fine_edges = find_fine_edges_in_coarse_edge(
                coarse_mesh, coarse_edge, fine_mesh, mesh_el
            )
            for fine_edge in fine_edges:
                v1, v2 = fine_mesh.edges[fine_edge.nr].vertices

                # get vertex DOFs
                coerse_el_edge_dofs.update(fes_fine.GetDofNrs(v1))
                coerse_el_edge_dofs.update(fes_fine.GetDofNrs(v2))

                # get edge coerse_el_edge_dofs
                coerse_el_edge_dofs.update(fes_fine.GetDofNrs(fine_edge))

    return coerse_el_edge_dofs


def get_coarse_element_interior(fine_mesh, coarse_el, fine_indices):
    """
    Get all fine elements that lie within a coarse element.
    Returns:
        fine_indices: set of fine element indices that lie within the coarse element.
    """
    interior_elements = []
    for fine_index in set(fine_indices):
        mesh_el = fine_mesh[ngs.ElementId(ngs.VOL, fine_index)]
        interior_elements.append(mesh_el)

    return interior_elements


# Example usage:
if __name__ == "__main__":
    lx, ly = 1.0, 1.0
    coarse_mesh_size = 0.15
    refinement_levels = 2
    fine_mesh, coarse_mesh = create_conforming_meshes(
        lx, ly, coarse_mesh_size, refinement_levels
    )
    print("Fine mesh:")
    print(f"\tNumber of elements: {fine_mesh.ne}")
    print(f"\tNumber of vertices: {fine_mesh.nv}")
    print(f"\tNumber of edges: {len(fine_mesh.edges)}")
    print("Coarse mesh:")
    print(f"\tNumber of elements: {coarse_mesh.ne}")
    print(f"\tNumber of vertices: {coarse_mesh.nv}")
    print(f"\tNumber of edges: {len(coarse_mesh.edges)}")
    fine_mesh.ngmesh.Save(str(DATA_DIR / "fine_mesh.vol"))
    coarse_mesh.ngmesh.Save(str(DATA_DIR / "coarse_mesh.vol"))
    vtk = ngs.VTKOutput(
        fine_mesh, coefs=[], names=[], filename=str(DATA_DIR / "fine_mesh")
    )
    vtk.Do()
    vtk = ngs.VTKOutput(
        coarse_mesh, coefs=[], names=[], filename=str(DATA_DIR / "coarse_mesh")
    )
    vtk.Do()

    # construct finite element space
    fes_fine = ngs.H1(fine_mesh, order=1, dirichlet="lid|floor|right|left")

    # construct coarse domains
    coarse_dict = find_fine_elements_within_coarse_elements(coarse_mesh, fine_mesh)

    # get fine degrees of freedom (DOFs)
    fine_dofs = set()
    for el in fes_fine.Elements(ngs.VOL):
        dofs = fes_fine.GetDofNrs(el)
        fine_dofs.update(dofs)

    # get coarse degrees of freedom (DOFs)
    coarse_dofs = set()
    for coarse_v in coarse_mesh.vertices:
        dofs = fes_fine.GetDofNrs(coarse_v)
        coarse_dofs.update(dofs)
    fine_dofs -= coarse_dofs

    # find all fine DOFs that lie on a coarse edge
    edge_dofs = set()
    fine_edge_count = 0
    figure, ax = plt.subplots(figsize=(8, 6))
    for i, (coarse_el, fine_indices) in enumerate(coarse_dict.items()):
        domain_elements = get_coarse_element_interior(
            fine_mesh, coarse_el, fine_indices
        )
        edge_dofs.update(get_coarse_element_edge_dofs(fine_mesh, coarse_el))
        neighbor_elements = extend_domain(fine_mesh, domain_elements)
        second_layer_elements = extend_domain(fine_mesh, neighbor_elements + domain_elements)
        if i % 8 == 0:
            fillcolor = domain_colors[i % len(domain_colors)]
            for el in domain_elements:
                plot_element(
                    ax,
                    el,
                    fine_mesh,
                    fillcolor=fillcolor,
                    alpha=0.7,
                    edgecolor="black",
                    label="Coarse Element",
                )
            for el in neighbor_elements:
                plot_element(
                    ax,
                    el,
                    fine_mesh,
                    fillcolor=fillcolor,
                    alpha=0.4,
                    edgecolor="black",
                    label="Overlap Element",
                )
            for el in second_layer_elements:
                plot_element(
                    ax,
                    el,
                    fine_mesh,
                    fillcolor=fillcolor,
                    alpha=0.2,
                    edgecolor="black",
                    label="Second Layer Element",
                )

    # remove coarse nodes from edge DOFs
    edge_dofs -= coarse_dofs

    # remove edge DOFs from fine DOFs
    fine_dofs -= edge_dofs

    print("FE space:")
    print(f"\t#DOFs: {fes_fine.ndof}")
    print(f"\t#Fine DOFs: {len(fine_dofs)}")
    print(f"\t#Edge DOFs: {len(edge_dofs)}")
    print(f"\t#Coarse DOFs: {len(coarse_dofs)}")
    plt.show()
