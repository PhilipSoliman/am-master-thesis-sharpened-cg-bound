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

from lib.eigenvalues import split_spectrum_into_clusters
from lib.logger import LOGGER
from lib.problems import CoefFunc
from lib.solvers import CustomCG
from lib.utils import get_cli_args, save_latex_figure, set_mpl_cycler

# get command line arguments
ARGS = get_cli_args()

# define coefficient functions to use
COEF_FUNCS = [
    CoefFunc.THREE_LAYER_VERTEX_INCLUSIONS,
    CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS,
]

# constants
FIGWIDTH = 3
FIGHEIGHT = 3
RECIPROCAL_COARSE_MESH_SIZES = [round(1 / mesh.coarse_mesh_size) for mesh in MESHES]
XTICKS = [rf"$\mathbf{{H = 1/{Nc}}}$" for Nc in RECIPROCAL_COARSE_MESH_SIZES]
XTICK_LOCS = np.arange(len(RECIPROCAL_COARSE_MESH_SIZES), dtype=int)

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

# to differentiate between actual iterations, classical and improved bounds
iter_colors = [None] * 3
iter_markers = [".", "x", "^"]
iter_linestyles = ["-", "--", "-."]

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
        niters_improved = []
        for mesh_params in MESHES:
            LOGGER.debug(f"Processing mesh H = {1/mesh_params.coarse_mesh_size:.0f}")

            # Load the eigenvalues from the saved numpy file
            fp = get_spectrum_save_path(
                mesh_params, coef_func, preconditioner_cls, coarse_space_cls
            )

            # check if the file exists
            if fp.exists():

                # load eigenvalues
                eigenvalues = np.load(fp)

                # number of iterations
                niters.append(len(eigenvalues))

                # get cluster coordinates
                clusters = split_spectrum_into_clusters(eigenvalues)
                LOGGER.debug(
                    (
                        f"Preconditioner {shorthand} has {len(clusters)} clusters:"
                        f"\n\t{[f'({c[0]:.2e}, {c[1]:.2e})' for c in clusters]}"
                        f"\n\tmin: {np.min(eigenvalues):.2e}, max: {np.max(eigenvalues):.2e}"
                    )
                )

                # get predicted number of iterations
                cond = np.abs(np.max(eigenvalues) / np.min(eigenvalues))
                niters_classical.append(
                    CustomCG.calculate_iteration_upperbound_static(
                        cond, log_rtol=np.log(RTOL), exact_convergence=False
                    )
                )

                # get improved number of iterations if applicable
                if len(clusters) >= 2:  # improved bound
                    niters_improved.append(
                        CustomCG.calculate_improved_cg_iteration_upperbound_static(
                            clusters, tol=RTOL, exact_convergence=False
                        )
                    )
                else:  # no improved bound
                    niters_improved.append(None)
                LOGGER.debug(
                    f"niters: {niters[-1]}, classical: {niters_classical[-1]}, improved: {niters_improved[-1] if niters_improved[-1] is not None else 'N/A'}"
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
            label="Iterations",
            linestyle=iter_linestyles[0],
        )

        # plot classical bound
        niters_classical_line = ax.plot(
            XTICK_LOCS,
            niters_classical,
            linestyle=iter_linestyles[1],
            marker=iter_markers[1],
            alpha=0.75,
            label="Classical Bound",
        )
        if iter_colors[1] is None:
            iter_colors[1] = niters_classical_line[0].get_color()
        else:
            niters_classical_line[0].set_color(iter_colors[1])

        # plot improved bound
        niters_improved_line = ax.plot(
            XTICK_LOCS,
            niters_improved,
            linestyle=iter_linestyles[2],
            marker=iter_markers[2],
            alpha=0.75,
            label="Improved Bound",
        )
        if iter_colors[2] is None:
            iter_colors[2] = niters_improved_line[0].get_color()
        else:
            niters_improved_line[0].set_color(iter_colors[2])

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
                ha="center",
                va="bottom",
                transform=ax.transAxes,
            )

    # add tick labels and xlabel to the last column axes
    if i == len(PRECONDITIONERS) - 1:
        for ax in precond_axs:
            ax.set_xticklabels(XTICKS, rotation=45, ha="right")

    # add ylabel and legend to the first column axes
    precond_axs[0].set_ylabel(shorthand, fontweight="bold")

    # add legend to the first plot
    if i == 0:
        precond_axs[0].legend()

# tight layout for the figure
fig.tight_layout()

if ARGS.generate_output:
    fn = Path(__file__).name.replace("_fig.py", "")
    save_latex_figure(fn, fig)
if ARGS.show_output:
    plt.show()
