from pathlib import Path
from typing import Type

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from approximate_spectra import (
    COEF_FUNCS,
    MESHES,
    PRECONDITIONERS,
    RTOL,
    get_spectrum_save_path,
)

from hcmsfem.cli import CLI_ARGS
from hcmsfem.logger import LOGGER
from hcmsfem.plot_utils import save_latex_figure, set_mpl_cycler
from hcmsfem.preconditioners import (
    AMSCoarseSpace,
    CoarseSpace,
    TwoLevelSchwarzPreconditioner,
)
from hcmsfem.problems import CoefFunc
from hcmsfem.solvers import (
    classic_cg_iteration_bound,
    multi_cluster_cg_iteration_bound,
    multi_tail_cluster_cg_iteration_bound,
)

# define coefficient functions to use
COEF_FUNCS = [
    CoefFunc.THREE_LAYER_VERTEX_INCLUSIONS,
    CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS,
]

# constants
FIGWIDTH = 3
FIGHEIGHT = 3
FONTSIZE = 9
LEGEND_SIZE = 0.3
RECIPROCAL_COARSE_MESH_SIZES = [round(1 / mesh.coarse_mesh_size) for mesh in MESHES]
XTICKS = [rf"$\mathbf{{H = 1/{Nc}}}$" for Nc in RECIPROCAL_COARSE_MESH_SIZES]
XTICK_LOCS = np.arange(len(RECIPROCAL_COARSE_MESH_SIZES), dtype=int)
LOG_RTOL = np.log(RTOL)

# set matplotlib cycler
set_mpl_cycler(lines=True, colors=True, markers=True)


def plot_absolute_performance(
    preconditioner_cls: Type[TwoLevelSchwarzPreconditioner],
    coarse_space_cls: Type[CoarseSpace],
    legend: bool = False,
):
    # initialize figure and axes for legend
    fig = plt.figure(figsize=(FIGWIDTH * len(COEF_FUNCS), FIGHEIGHT + LEGEND_SIZE))
    gs = gridspec.GridSpec(
        2, len(COEF_FUNCS), height_ratios=[1, LEGEND_SIZE / FIGHEIGHT]
    )
    axs = []
    for i in range(len(COEF_FUNCS)):
        axs.append(
            fig.add_subplot(gs[0, i], sharey=None if i == 0 else axs[0])
        )
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    # to differentiate between actual iterations, classical and sharpened bounds
    iter_colors = [None] * 4
    iter_markers = [".", "x", "^", "o"]
    # Use visually distinct linestyles: solid, dashed, dotted, dash-dot-dot
    iter_linestyles = ["-", "--", ":", "-."]

    # preconditioner shorthand
    shorthand = f"{preconditioner_cls.SHORT_NAME}-{coarse_space_cls.SHORT_NAME}"

    for i, coef_func in enumerate(COEF_FUNCS):
        LOGGER.debug(f"Processing coefficient function: {coef_func.short_name}")
        niters = []
        niters_classical = []
        niters_multi_cluster = []
        niters_tail_cluster = []
        for mesh_params in MESHES:
            LOGGER.debug(f"Processing mesh H = {1/mesh_params.coarse_mesh_size:.0f}")

            # Load the eigenvalues from the saved numpy file
            fp = get_spectrum_save_path(
                mesh_params, coef_func, preconditioner_cls, coarse_space_cls
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

                # get tail-cluster bound
                niters_tail_cluster.append(
                    multi_tail_cluster_cg_iteration_bound(
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

        # get the axes for the preconditioner
        ax = axs[i]

        # plot iterations & bounds
        niters_line = ax.plot(
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
            label="$m_{N_{\\mathrm{cluster}}}$",
        )
        if iter_colors[2] is None:
            iter_colors[2] = niters_multi_cluster_line[0].get_color()
        else:
            niters_multi_cluster_line[0].set_color(iter_colors[2])

        # plot tail-cluster bound
        niters_tail_cluster_line = ax.plot(
            XTICK_LOCS,
            niters_tail_cluster,
            linestyle=iter_linestyles[3],
            marker=iter_markers[3],
            alpha=0.75,
            label="$m_{N_{\\mathrm{tail-cluster}}}$",
        )
        if iter_colors[3] is None:
            iter_colors[3] = niters_tail_cluster_line[0].get_color()
        else:
            niters_tail_cluster_line[0].set_color(iter_colors[3])

        # format the axes (all)
        ax.grid()
        ax.set_xticks(XTICK_LOCS)
        ax.set_yscale("log")

    # add title to top row axes
    for i, ax in enumerate(axs):
        ax.text(
            0.5,
            1.0,
            COEF_FUNCS[i].latex,
            fontweight="bold",
            fontsize=FONTSIZE,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )

    # add tick labels and xlabel to the last column axes
    for ax in axs:
        ax.set_xticklabels(XTICKS, rotation=45, ha="right", fontsize=8)

    # add ylabel and legend to the first column axes
    axs[0].set_ylabel(shorthand, fontweight="bold", fontsize=FONTSIZE)

    # add legend to the bottom axis
    if legend:
        handles, labels = axs[0].get_legend_handles_labels()
        legend_ax.legend(
            handles,
            labels,
            fontsize=FONTSIZE,
            loc="center",
            ncol=len(labels),
            frameon=False,
        )

    # tight layout for the figure
    fig.tight_layout()

    return fig


# main loop
if __name__ == "__main__":
    for i, ((preconditioner_cls, coarse_space_cls)) in enumerate(PRECONDITIONERS):
        legend = True
        # if coarse_space_cls == AMSCoarseSpace:
        #     legend = True
        fig = plot_absolute_performance(
            preconditioner_cls,
            coarse_space_cls,
            legend=legend,
        )
        shorthand = f"{preconditioner_cls.SHORT_NAME}-{coarse_space_cls.SHORT_NAME}"
        if CLI_ARGS.generate_output:
            fn = Path(__file__).name.replace("_fig.py", f"_{shorthand}")
            save_latex_figure(fn, fig)
    if CLI_ARGS.show_output:
        plt.show()
