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

from hcmsfem.cli import get_cli_args
from hcmsfem.eigenvalues import eigs
from hcmsfem.logger import LOGGER
from hcmsfem.plot_utils import save_latex_figure, set_mpl_cycler, set_mpl_style
from hcmsfem.solvers import CustomCG, sharpened_cg_iteration_bound

# set matplotlib style & cycler
set_mpl_style()
set_mpl_cycler(colors=True)

# get cli args
ARGS = get_cli_args()
FIGWIDTH = 5
FIGHEIGHT = 2

# tolerance
LOG_RTOL = np.log(RTOL)

# initialize figure and axes
fig, axs = plt.subplots(
    len(COEF_FUNCS),
    len(MESHES),
    figsize=(FIGWIDTH * len(MESHES), FIGHEIGHT * len(COEF_FUNCS)),
    squeeze=False,
    sharex=True,
    sharey=True,
)

# GOAL: get an idea of how fast the sharpened bound converges to its final prediction of the number of iterations
# TODO 1: rerun approximate_spectra.py to get alpha and beta cg arrays: DONE

# main plot loop
for i, mesh_params in enumerate(MESHES):
    axes = axs[:, i]
    for coef_func, ax in zip(COEF_FUNCS, axes):
        niters = {}
        for preconditioner_cls, coarse_space_cls in PRECONDITIONERS:
            fp = get_spectrum_save_path(
                mesh_params, coef_func, preconditioner_cls, coarse_space_cls
            )
            if fp.exists():
                # Load the alpha and beta arrays from the saved numpy file
                array_zip = np.load(fp)
                alpha = array_zip["alpha"]
                beta = array_zip["beta"]

                # get shorthand
                shorthand = (
                    f"{preconditioner_cls.SHORT_NAME}-{coarse_space_cls.SHORT_NAME}"
                )

                # loop over iterations
                niters[shorthand] = np.zeros(len(alpha), dtype=int)
                for j in range(len(alpha)):
                    # # TODO 2: for every coefficient function, mesh size and preconditioner, calculate the eigenvalues
                    lanczos_matrix = CustomCG.get_lanczos_matrix_from_coefficients(
                        alpha[: j + 1], beta[:j]
                    )

                    # TODO 3: feed eigenvalues to sharpened_cg_iteration_bound...
                    eigenvalues = eigs(lanczos_matrix)

                    # apply sharpened bound
                    niters[shorthand][i] = sharpened_cg_iteration_bound(
                        eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                    )
            else:
                # Provide a clickable link to the script in the repo using Rich markup with absolute path
                approx_path = Path(__file__).parent / "approximate_spectra.py"
                LOGGER.error(
                    f"File %s does not exist. Run '[link=file:{approx_path}]approximate_spectra.py[/link]' first.",
                    fp,
                )
                exit()

        # TODO 3 (continued): ...and plot the number of iterations it calculates
        for shorthand, iterations in niters.items():
            ax.plot(
                range(len(iterations)),
                iterations,
                label=shorthand,
                marker="o",
                markersize=3,
            )

            # TODO 4: plot a horizontal line for the number of iterations that the sharpened bound predicts for the eigenspectrum at convergence
            ax.axhline(
                iterations[-1],
                linestyle="--",
                color=ax.lines[-1].get_color(),
                linewidth=0.8,
            )

        # ax.set_yscale("log")

        # grid settings
        ax.grid(axis="x", which="both", linestyle="--", linewidth=0.7)
        ax.grid(axis="y", which="both", linestyle=":", linewidth=0.5)

# Add column titles (LaTeX, Nc as integer, no bold for compatibility)
for col_idx, mesh_params in enumerate(MESHES):
    H = mesh_params.coarse_mesh_size
    Nc = int(1 / H)
    ax = axs[0, col_idx] if hasattr(axs, "ndim") and axs.ndim == 2 else axs[0]
    ax.set_title(rf"$\mathbf{{H = 1/{Nc}}}$", fontsize=11)

# Add row labels (rotated, bold, fontsize 9) at the beginning of each row
for row_idx, coef_func in enumerate(COEF_FUNCS):
    # Get the y-position as the center of the row of axes
    ax = axs[row_idx, 0]

    # Use axes coordinates to place the text just outside the left of the axes
    fig.text(
        0,  # x-position (fraction of figure width, adjust as needed)
        ax.get_position().y0
        + ax.get_position().height / 2,  # y-position (center of the row)
        coef_func.latex,  # use LaTeX representation
        va="center",
        ha="left",
        rotation=90,
        fontweight="bold",
        fontsize=14,
    )

# tight layout for the figure
fig.tight_layout(pad=1.3)

if ARGS.generate_output:
    fn = Path(__file__).name.replace("_fig.py", "")
    save_latex_figure(fn, fig)
if ARGS.show_output:
    plt.show()
