import subprocess
from os import remove
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import ngsolve as ngs
import numpy as np
from matplotlib.pyplot import savefig

from hcmsfem.boundary_conditions import HomogeneousDirichlet
from hcmsfem.cli import get_cli_args
from hcmsfem.meshes import DefaultQuadMeshParams, TwoLevelMesh
from hcmsfem.plot_utils import FIG_FOLDER, LATEX_STANDALONE_PGF_PRE, CustomColors, set_mpl_style
from hcmsfem.problem_type import ProblemType
from hcmsfem.problems import Problem

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


if __name__ == "__main__":
    two_mesh_4 = TwoLevelMesh(mesh_params=DefaultQuadMeshParams.Nc4)
    figs = [plot_edge_inclusions(two_mesh_4)]
    fns = ["edge_inclusions"]
    for fig, fn in zip(figs, fns):
        fig.tight_layout()
        if CLI_ARGS.generate_output:
            fp = Path(__file__).name.replace("_fig.py", f"_{fn}")
            save_latex_figure(fp, fig)
    if CLI_ARGS.show_output:
        plt.show()
