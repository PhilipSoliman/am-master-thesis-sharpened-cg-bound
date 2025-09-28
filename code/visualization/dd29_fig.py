import subprocess
import sys
from os import remove
from pathlib import Path
from typing import Type

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import ngsolve as ngs
import numpy as np
from matplotlib.pyplot import savefig

from hcmsfem.boundary_conditions import HomogeneousDirichlet
from hcmsfem.cli import get_cli_args
from hcmsfem.logger import LOGGER
from hcmsfem.meshes import DefaultQuadMeshParams, TwoLevelMesh
from hcmsfem.plot_utils import FIG_FOLDER, CustomColors, set_mpl_cycler, set_mpl_style
from hcmsfem.preconditioners import (
    AMSCoarseSpace,
    CoarseSpace,
    GDSWCoarseSpace,
    RGDSWCoarseSpace,
    TwoLevelSchwarzPreconditioner,
)
from hcmsfem.problem_type import ProblemType
from hcmsfem.problems import CoefFunc, Problem
from hcmsfem.root import get_venv_root
from hcmsfem.solvers import classic_cg_iteration_bound, multi_cluster_cg_iteration_bound

sys.path.append((get_venv_root() / "code" / "model_spectra").as_posix())
from approximate_spectra import get_spectrum_save_path  # type: ignore

CLI_ARGS = get_cli_args()
LATEX_STANDALONE_DOCUMENTCLASS = r"\documentclass{standalone}"

LATEX_STANDALONE_PREAMBLE = r"""
\def\mathdefault#1{#1}
\everymath=\expandafter{\the\everymath\displaystyle}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{siunitx}
\usepackage{pgf}
\usepackage{lmodern}
\usepackage{newtxtext}
\usepackage[varvw]{newtxmath}
\makeatletter\@ifpackageloaded{underscore}{}{\usepackage[strings]{underscore}}\makeatother
"""

LATEX_STANDALONE_BEGIN = r"""
\begin{document} 
"""

LATEX_STANDALONE_END = r""" 
\end{document}
"""

FONTSIZE = 10  # in pt

CONTRAST_COLOR = CustomColors.RED.value
BACKGROUND_COLOR = CustomColors.NAVY.value


def set_mpl_style(fontsize: int = FONTSIZE):
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.preamble": LATEX_STANDALONE_PREAMBLE,
            "axes.labelweight": "bold",
            "lines.linewidth": 1.5,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "legend.frameon": True,
            "legend.framealpha": 1,
            "legend.fancybox": True,
            "legend.shadow": True,
            "legend.borderpad": 1,
            "legend.borderaxespad": 1,
            "legend.handletextpad": 1,
            "legend.handlelength": 1.5,
            "legend.labelspacing": 1,
            "legend.columnspacing": 2,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "savefig.transparent": False,
            "savefig.orientation": "landscape",
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "legend.title_fontsize": fontsize,
            "figure.titlesize": fontsize,
        }
    )


def save_latex_figure(fn: str, fig: plt.Figure | None = None) -> None:
    # ensure figure directory exists
    FIG_FOLDER.mkdir(parents=True, exist_ok=True)

    # save currently active or provided figure as pgf using matplotlib
    pgf_p = FIG_FOLDER / (fn + ".pgf")
    if fig is None:
        savefig(pgf_p, backend="pgf")
    else:
        fig.savefig(pgf_p, backend="pgf")

    # create tex file
    tex_p = pgf_p.with_suffix(".tex")
    with open(tex_p, "w") as tex_f:
        tex_f.write(LATEX_STANDALONE_DOCUMENTCLASS)
        tex_f.write(LATEX_STANDALONE_PREAMBLE)
        tex_f.write(LATEX_STANDALONE_BEGIN)
        tex_f.write(r"    \input{" + pgf_p.as_posix() + r"}")
        tex_f.write(LATEX_STANDALONE_END)

    # run lualatex on tex file
    subprocess.run(
        ["lualatex", tex_p.name], check=True, cwd=FIG_FOLDER, capture_output=True
    )
    print(f"Saved figure to {tex_p.with_suffix('.pdf')}")

    # clean up .aux and .log files
    for file in FIG_FOLDER.glob("*.aux"):
        remove(file)
    for file in FIG_FOLDER.glob("*.log"):
        remove(file)

    # remove tex and pgf files
    remove(tex_p)
    remove(pgf_p)


def plot_edge_inclusions(two_mesh: TwoLevelMesh) -> plt.Figure:
    set_mpl_style()
    fig, ax_edgeslabs = plt.subplots(
        1, 1, figsize=(3, 2.5), squeeze=True, sharex=True, sharey=True
    )

    # plot coarse mesh on both axes
    two_mesh.plot_mesh(ax_edgeslabs, mesh_type="coarse")

    # instantiate diffusion problem
    problem = Problem(
        [HomogeneousDirichlet(ProblemType.DIFFUSION)],
        mesh=two_mesh,
        ptype=ProblemType.DIFFUSION,
    )

    # construct fespace
    problem.construct_fespace()

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
    all_elements = np.array([el.nr for el in two_mesh.fine_mesh.Elements()])
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

    return fig


# PCG iteration constants
COEF_FUNC = CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS
RTOL = 1e-8
LOG_RTOL = np.log(RTOL)
MESHES = DefaultQuadMeshParams

# plot
FIGWIDTH = 1.5
FIGHEIGHT = 1.5
FONTSIZE = 9
LEGEND_SIZE = 0.56
RECIPROCAL_COARSE_MESH_SIZES = [round(1 / mesh.coarse_mesh_size) for mesh in MESHES]
XTICKS = [rf"$\mathbf{{H = 1/{Nc}}}$" for Nc in RECIPROCAL_COARSE_MESH_SIZES]
XTICK_LOCS = np.arange(len(RECIPROCAL_COARSE_MESH_SIZES), dtype=int)
PADDING = dict(hspace=0.01, wspace=0.1, left=0.08, right=0.99, top=0.93, bottom=0)
set_mpl_cycler(lines=True, colors=True, markers=True)


def plot_absolute_performance(
    coarse_spaces: list[Type[CoarseSpace]],
    legend: bool = False,
):
    # initialize figure and axes for legend
    fig = plt.figure(
        figsize=(
            FIGWIDTH * len(coarse_spaces),
            FIGHEIGHT  + LEGEND_SIZE,
        )
    )
    total_height = FIGHEIGHT + LEGEND_SIZE
    gs = gridspec.GridSpec(
        2,
        len(coarse_spaces),
        height_ratios=[
            FIGHEIGHT / total_height,
            LEGEND_SIZE / total_height,
        ],
    )
    axs = []
    for i in range(len(coarse_spaces)):
        axs.append(fig.add_subplot(gs[0, i], sharey=None if i == 0 else axs[0]))
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")
    fig.subplots_adjust(**PADDING)

    # to differentiate between actual iterations, classical and sharpened bounds
    iter_colors = [None] * len(coarse_spaces)
    iter_markers = [".", "x", "^"]  # = len(coarse_spaces)

    # Use visually distinct linestyles: solid, dashed, dotted, dash-dot-dot
    iter_linestyles = ["-", "--", ":"]  # = len(coarse_spaces)

    for i, coarse_space_cls in enumerate(coarse_spaces):
        LOGGER.debug(f"Processing coefficient function: {coarse_space_cls.SHORT_NAME}")

        niters = []
        niters_classical = []
        niters_multi_cluster = []
        for mesh_params in MESHES:
            LOGGER.debug(f"Processing mesh H = {1/mesh_params.coarse_mesh_size:.0f}")

            # Load the eigenvalues from the saved numpy file
            fp = get_spectrum_save_path(
                mesh_params, COEF_FUNC, TwoLevelSchwarzPreconditioner, coarse_space_cls
            )

            # check if the file exists
            if fp.exists():

                # load eigenvalues
                eigenvalues = np.load(fp)["eigenvalues"]

                # number of iterations
                niters.append(len(eigenvalues))

                # get predicted number of iterations
                cond = np.abs(np.max(eigenvalues) / np.min(eigenvalues))
                niters_classical.append(
                    classic_cg_iteration_bound(
                        cond, log_rtol=LOG_RTOL, exact_convergence=False
                    )
                )

                # get multi-cluster bound
                niters_multi_cluster.append(
                    multi_cluster_cg_iteration_bound(
                        eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                    )
                )

                LOGGER.debug(
                    f"niters: {niters[-1]}, classical: {niters_classical[-1]}, improved: {niters_multi_cluster[-1] if niters_multi_cluster[-1] is not None else 'N/A'}"
                )

            else:
                # Provide a clickable link to the script in the repo using Rich markup with absolute path
                approx_path = Path(__file__).parent / "approximate_spectra.py"
                LOGGER.error(
                    f"File %s does not exist. Run '[link=file:{approx_path}]approximate_spectra.py[/link]' first.",
                    fp,
                )
                exit()

        # get the axes for the coarse space
        ax = axs[i]

        # plot iterations & bounds
        ax.plot(
            XTICK_LOCS,
            niters,
            label="$m$",
            linestyle=iter_linestyles[0],
        )

        # plot classical bound
        niters_classical_line = ax.plot(
            XTICK_LOCS,
            niters_classical,
            linestyle=iter_linestyles[1],
            marker=iter_markers[1],
            alpha=0.75,
            label="$m_1$",
        )
        if iter_colors[1] is None:
            iter_colors[1] = niters_classical_line[0].get_color()
        else:
            niters_classical_line[0].set_color(iter_colors[1])

        # plot multi-cluster bound
        niters_multi_cluster_line = ax.plot(
            XTICK_LOCS,
            niters_multi_cluster,
            linestyle=iter_linestyles[2],
            marker=iter_markers[2],
            alpha=0.75,
            label="$m_s$",
        )
        if iter_colors[2] is None:
            iter_colors[2] = niters_multi_cluster_line[0].get_color()
        else:
            niters_multi_cluster_line[0].set_color(iter_colors[2])

        # format the axes (all)
        ax.grid()
        ax.set_xticks(XTICK_LOCS)
        ax.set_yscale("log")
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.3)  # Minor grid with lower alpha
        ax.grid(True, which="major", alpha=0.7)  # Major grid with higher alpha

        # Hide tick labels for all axes except the first one
        if i > 0:
            ax.tick_params(bottom=True, labelbottom=False, left=True, labelleft=False)

    # add title to top row axes
    for i, ax in enumerate(axs):
        ax.text(
            0.5,
            1.01,
            coarse_spaces[i].SHORT_NAME,
            fontweight="bold",
            fontsize=FONTSIZE,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )

    # add tick labels and xlabel to the last column axes
    # for ax in axs:
    axs[0].set_xticklabels(XTICKS, rotation=45, ha="right", fontsize=8)

    # add legend to the bottom axis
    if legend:
        handles, labels = axs[0].get_legend_handles_labels()
        legend_ax.legend(
            handles,
            labels,
            fontsize=FONTSIZE,
            loc="center right",
            ncol=len(labels),
            frameon=False,
        )

    return fig


if __name__ == "__main__":
    two_mesh_4 = TwoLevelMesh(mesh_params=DefaultQuadMeshParams.Nc4)
    coarse_spaces = [AMSCoarseSpace, GDSWCoarseSpace, RGDSWCoarseSpace]
    figs = [
        # plot_edge_inclusions(two_mesh_4).tight_layout(),
        plot_absolute_performance(coarse_spaces, legend=True),
    ]
    fns = [
        # "edge_inclusions",
        "absolute_performance"
    ]
    for fig, fn in zip(figs, fns):
        if CLI_ARGS.generate_output:
            fp = Path(__file__).name.replace("_fig.py", f"_{fn}")
            save_latex_figure(fp, fig)
    if CLI_ARGS.show_output:
        plt.show()
        plt.show()
