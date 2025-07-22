from pathlib import Path

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
from hcmsfem.problems import CoefFunc
from hcmsfem.solvers import (
    classic_cg_iteration_bound,
    mixed_sharpened_cg_iteration_bound,
    sharpened_cg_iteration_bound,
)

# define coefficient functions to use
COEF_FUNCS = [
    CoefFunc.THREE_LAYER_VERTEX_INCLUSIONS,
    CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS,
]

# constants
FIGWIDTH = 2
FIGHEIGHT = 2
FONTSIZE = 9
RECIPROCAL_COARSE_MESH_SIZES = [round(1 / mesh.coarse_mesh_size) for mesh in MESHES]
XTICKS = [rf"$\mathbf{{H = 1/{Nc}}}$" for Nc in RECIPROCAL_COARSE_MESH_SIZES]
XTICK_LOCS = np.arange(len(RECIPROCAL_COARSE_MESH_SIZES), dtype=int)
LOG_RTOL = np.log(RTOL)

# set matplotlib cycler
set_mpl_cycler(lines=True, colors=True, markers=True)

# initialize figure and axes
fig, axs = plt.subplots(
    len(PRECONDITIONERS),
    len(COEF_FUNCS),
    figsize=(FIGWIDTH * len(COEF_FUNCS), FIGHEIGHT * len(PRECONDITIONERS)),
    squeeze=False,
    sharex=True,
    sharey=True,
)

# to differentiate between actual iterations, classical and sharpened bounds
iter_colors = [None] * 4
iter_markers = [".", "x", "^", "o"]
iter_linestyles = ["-", "--", "-.", ":"]

# main loop
for i, ((preconditioner_cls, coarse_space_cls), precond_axs) in enumerate(
    zip(PRECONDITIONERS, axs)
):
    # preconditioner shorthand
    shorthand = f"{preconditioner_cls.SHORT_NAME}-{coarse_space_cls.SHORT_NAME}"

    for j, coef_func in enumerate(COEF_FUNCS):
        LOGGER.debug(f"Processing coefficient function: {coef_func.short_name}")
        niters = []
        niters_classical = []
        niters_sharpened = []
        niters_sharpened_mixed = []
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

                # get sharpened bound
                niters_sharpened.append(
                    sharpened_cg_iteration_bound(
                        eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                    )
                )

                # get mixed sharpened bound
                niters_sharpened_mixed.append(
                    mixed_sharpened_cg_iteration_bound(
                        eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                    )
                )

                LOGGER.debug(
                    f"niters: {niters[-1]}, classical: {niters_classical[-1]}, improved: {niters_sharpened[-1] if niters_sharpened[-1] is not None else 'N/A'}"
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
        ax = precond_axs[j]

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

        # plot improved bound
        niters_sharpened_line = ax.plot(
            XTICK_LOCS,
            niters_sharpened,
            linestyle=iter_linestyles[2],
            marker=iter_markers[2],
            alpha=0.75,
            label="$m_{N_{\\mathrm{cluster}}}$",
        )
        if iter_colors[2] is None:
            iter_colors[2] = niters_sharpened_line[0].get_color()
        else:
            niters_sharpened_line[0].set_color(iter_colors[2])

        # plot mixed sharpened bound
        niters_sharpened_mixed_line = ax.plot(
            XTICK_LOCS,
            niters_sharpened_mixed,
            linestyle=iter_linestyles[2],
            marker=iter_markers[2],
            alpha=0.75,
            label="$m_{N_{\\mathrm{tail-cluster}}}$",
        )
        if iter_colors[3] is None:
            iter_colors[3] = niters_sharpened_mixed_line[0].get_color()
        else:
            niters_sharpened_mixed_line[0].set_color(iter_colors[3])

        # format the axes (all)
        ax.grid()
        ax.set_xticks(XTICK_LOCS)
        ax.set_yscale("log")

    # add title to top row axes
    if i == 0:
        for j, ax in enumerate(precond_axs):
            ax.text(
                0.5,
                1.0,
                COEF_FUNCS[j].latex,
                fontweight="bold",
                fontsize=FONTSIZE,
                ha="center",
                va="bottom",
                transform=ax.transAxes,
            )

    # add tick labels and xlabel to the last column axes
    if i == len(PRECONDITIONERS) - 1:
        for ax in precond_axs:
            ax.set_xticklabels(XTICKS, rotation=45, ha="right")

    # add ylabel and legend to the first column axes
    precond_axs[0].set_ylabel(shorthand, fontweight="bold", fontsize=FONTSIZE)

    # add legend to the first plot
    if i == 2:
        precond_axs[0].legend(fontsize=FONTSIZE, loc="center left")

# tight layout for the figure
fig.tight_layout()

if CLI_ARGS.generate_output:
    fn = Path(__file__).name.replace("_fig.py", "")
    save_latex_figure(fn, fig)
if CLI_ARGS.show_output:
    plt.show()
