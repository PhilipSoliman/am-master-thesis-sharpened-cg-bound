from pathlib import Path

import matplotlib.pyplot as plt
import ngsolve as ngs
import numpy as np

from hcmsfem.boundary_conditions import HomogeneousDirichlet
from hcmsfem.cli import CLI_ARGS
from hcmsfem.meshes import DefaultQuadMeshParams, TwoLevelMesh
from hcmsfem.plot_utils import CustomColors, save_latex_figure, set_mpl_style
from hcmsfem.problem_type import ProblemType
from hcmsfem.problems import CoefFunc, Problem, SourceFunc

FIGWIDTH = 3.0
FIGHEIGHT = 3.0
CONTRAST_COLOR = CustomColors.RED.value
BACKGROUND_COLOR = CustomColors.NAVY.value


def plot_meshes_and_domains(two_mesh: TwoLevelMesh) -> plt.Figure:
    """
    Plot the two-level mesh.
    """
    set_mpl_style()
    fig, (mesh_ax, domains_ax) = plt.subplots(
        1, 2, figsize=(2 * FIGWIDTH, FIGHEIGHT), squeeze=True, sharex=True, sharey=True
    )
    two_mesh.plot_mesh(mesh_ax, mesh_type="fine")
    two_mesh.plot_mesh(mesh_ax, mesh_type="coarse")
    mesh_ax.set_title("$Q_h$ and $Q_H$")

    two_mesh.plot_domains(domains_ax, domains=3, plot_layers=True)
    domains_ax.set_title("$\\Omega_i$")
    return fig


def plot_coefficient_functions(two_mesh: TwoLevelMesh) -> plt.Figure:
    """
    Plot the edge slabs of the two-level mesh.
    """
    set_mpl_style()
    fig, (ax_3layervertices, ax_edgeslabs) = plt.subplots(
        1, 2, figsize=(2 * FIGWIDTH, FIGHEIGHT), squeeze=True, sharex=True, sharey=True
    )

    # plot coarse mesh on both axes
    two_mesh.plot_mesh(ax_3layervertices, mesh_type="coarse")
    two_mesh.plot_mesh(ax_edgeslabs, mesh_type="coarse")

    # instantiate diffusion problem
    problem = Problem(
        [HomogeneousDirichlet(ProblemType.DIFFUSION)],
        mesh=two_mesh,
        ptype=ProblemType.DIFFUSION,
    )

    # construct fespace
    problem.construct_fespace()

    ######################
    # three layer vertex #
    ######################
    free_coarse_nodes = list(problem.fes.free_component_tree_dofs.keys())

    # loop over coarse nodes and set high contrast to 3 layers of elements around them
    contrast_elements = []
    for coarse_node in free_coarse_nodes:
        mesh_el = two_mesh.fine_mesh[coarse_node]
        elements_first_layer = set(mesh_el.elements)
        for element_first_layer in elements_first_layer:
            contrast_elements.append(element_first_layer.nr)

            # get second layer elements
            outer_vertices_first_layer = set(
                two_mesh.fine_mesh[element_first_layer].vertices
            ) - set([coarse_node])
            for outer_vertex_first_layer in outer_vertices_first_layer:
                # filter inner elements
                elements_second_layer = (
                    set(two_mesh.fine_mesh[outer_vertex_first_layer].elements)
                    - elements_first_layer
                )
                for element_second_layer in elements_second_layer:
                    # add second layer
                    contrast_elements.append(element_second_layer.nr)

                    # get third layer elements
                    outer_vertices_second_layer = (
                        set(two_mesh.fine_mesh[element_second_layer].vertices)
                        - outer_vertices_first_layer
                        - set([coarse_node])
                    )
                    for outer_vertex_second_layer in outer_vertices_second_layer:
                        # filter inner elements
                        elements_third_layer = (
                            set(two_mesh.fine_mesh[outer_vertex_second_layer].elements)
                            - elements_first_layer
                            - elements_second_layer
                        )
                        for element_third_layer in elements_third_layer:
                            # add third layer
                            contrast_elements.append(element_third_layer.nr)

    for el_nr in contrast_elements:
        two_mesh.plot_element(
            ax_3layervertices,
            two_mesh.fine_mesh[ngs.ElementId(el_nr)],
            two_mesh.fine_mesh,
            fillcolor=CONTRAST_COLOR,
            edgecolor="black",
            linewidth=0.5,
            alpha=1.0,
        )
    all_elements = np.array([el.nr for el in two_mesh.fine_mesh.Elements()])
    background_elements = np.setdiff1d(
        all_elements, contrast_elements, assume_unique=True
    )
    for el_nr in background_elements:
        two_mesh.plot_element(
            ax_3layervertices,
            two_mesh.fine_mesh[ngs.ElementId(el_nr)],
            two_mesh.fine_mesh,
            fillcolor=BACKGROUND_COLOR,
            edgecolor=BACKGROUND_COLOR,
            linewidth=0.1,
            alpha=0.9,
        )
    ax_3layervertices.set_title(CoefFunc.THREE_LAYER_VERTEX_INCLUSIONS.latex)

    ##############################
    # edge slabs around vertices #
    ##############################
    contrast_elements = []
    for coarse_node in problem.fes.free_component_tree_dofs.keys():
        slab_elements = two_mesh.edge_slabs["around_coarse_nodes"][coarse_node.nr]
        contrast_elements.extend(slab_elements)
    for el_nr in contrast_elements:
        two_mesh.plot_element(
            ax_edgeslabs,
            two_mesh.fine_mesh[ngs.ElementId(el_nr)],
            two_mesh.fine_mesh,
            fillcolor=CONTRAST_COLOR,
            edgecolor="black",
            linewidth=0.5,
            alpha=1.0,
        )
    background_elements = np.setdiff1d(
        all_elements, contrast_elements, assume_unique=True
    )
    for el_nr in background_elements:
        two_mesh.plot_element(
            ax_edgeslabs,
            two_mesh.fine_mesh[ngs.ElementId(el_nr)],
            two_mesh.fine_mesh,
            fillcolor=BACKGROUND_COLOR,
            edgecolor=BACKGROUND_COLOR,
            linewidth=0.1,
            alpha=0.9,
        )
    ax_edgeslabs.set_title(CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS.latex)

    return fig


if __name__ == "__main__":
    two_mesh = TwoLevelMesh(mesh_params=DefaultQuadMeshParams.Nc4)
    figs = [plot_meshes_and_domains(two_mesh), plot_coefficient_functions(two_mesh)]
    fns = [
        "meshes_and_domains",
        CoefFunc.THREE_LAYER_VERTEX_INCLUSIONS.short_name,
        CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS.short_name,
    ]
    for fig, fn in zip(figs, fns):
        fig.tight_layout()
        if CLI_ARGS.generate_output:
            fp = Path(__file__).name.replace("_fig.py", f"_{fn}")
            save_latex_figure(fp, fig)
    if CLI_ARGS.show_output:
        plt.show()
