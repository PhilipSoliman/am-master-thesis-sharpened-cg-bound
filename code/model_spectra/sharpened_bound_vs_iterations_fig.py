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
from sharpened_bound_vs_iterations import N_ITERATIONS

from hcmsfem.cli import get_cli_args
from hcmsfem.logger import LOGGER, PROGRESS
from hcmsfem.meshes import DefaultQuadMeshParams
from hcmsfem.plot_utils import save_latex_figure, set_mpl_cycler, set_mpl_style
from hcmsfem.solvers import mixed_sharpened_cg_iteration_bound

PLOT_MESHES = [DefaultQuadMeshParams.Nc8, DefaultQuadMeshParams.Nc64]
SHOW_BOUNDS = False

# set matplotlib style & cycler
set_mpl_style()
set_mpl_cycler(colors=True)

# get cli args
ARGS = get_cli_args()
FIGWIDTH = 5
FIGHEIGHT = 2

# tolerance
LOG_RTOL = np.log(RTOL)

# number of iterations to calculate
N_ITERATIONS = 500
MOVING_AVG_WINDOW = 10


def plot_sharpened_bound_vs_iterations(
    preconditioner, meshes=MESHES, coef_funcs=COEF_FUNCS, show_bounds=True
) -> tuple[plt.Figure, str]:
    # get preconditioner class and coarse space class
    preconditioner_cls, coarse_space_cls = preconditioner

    # get shorthand
    shorthand = f"{preconditioner_cls.SHORT_NAME}-{coarse_space_cls.SHORT_NAME}"

    # initialize figure and axes
    fig, axs = plt.subplots(
        len(coef_funcs),
        len(meshes),
        figsize=(FIGWIDTH * len(meshes), FIGHEIGHT * len(coef_funcs)),
        squeeze=False,
        sharey=True,
    )

    # initialize progress bar
    progress = PROGRESS.get_active_progress_bar()
    main_task = progress.add_task(
        "Calculating upper bound vs iterations", total=len(meshes)
    )
    main_desc = progress.get_description(main_task)
    main_desc += " ([bold]H = 1/{0:.0f}, CF = {1}[/bold], M = {2})"

    # main plot loop
    for i, mesh_params in enumerate(meshes):
        for coef_func, ax in zip(coef_funcs, axs[:, i]):
            fp = get_spectrum_save_path(
                mesh_params, coef_func, preconditioner_cls, coarse_space_cls
            )
            if fp.exists():
                # Load the alpha and beta arrays from the saved numpy file
                array_zip = np.load(fp)
                niters_sharp = array_zip["niters_sharp"]
                niters_sharp_mixed = array_zip["niters_sharp_mixed"]

                # eigenvalues at convergence
                convergence_eigenvalues = array_zip["eigenvalues"]
            else:
                approx_path = Path(__file__).parent / "sharpened_bound_vs_iterations.py"
                LOGGER.error(
                    f"File %s does not exist. Run '[link=file:{approx_path}]sharpened_bound_vs_iterations.py[/link]' first.",
                    fp,
                )
                exit()

            plot_bounds(
                ax,
                shorthand,
                convergence_eigenvalues,
                niters_sharp,
                niters_sharp_mixed,
                show_bounds,
            )

        # advance main task
        progress.advance(main_task)

    style_figure(fig, axs, shorthand, meshes, coef_funcs)

    # stop progress bar
    progress.soft_stop()

    return fig, shorthand


def plot_bounds(
    ax,
    shorthand,
    convergence_eigenvalues,
    niters_sharp,
    niters_sharp_mixed,
    show_bounds: bool,
):
    # get number of iterations to plot
    n_iters_plot = min(len(convergence_eigenvalues) - 1, N_ITERATIONS)

    # plot the two versions of the sharpened bound
    sharp_iters = niters_sharp
    sharp_line = ax.plot(
        range(n_iters_plot),
        sharp_iters[:n_iters_plot],
        marker="v" if show_bounds else None,
        linestyle="None",
        markersize=3,
    )

    sharp_mixed_iters = niters_sharp_mixed
    sharp_mixed_line = ax.plot(
        range(n_iters_plot),
        sharp_mixed_iters[:n_iters_plot],
        marker="^" if show_bounds else None,
        linestyle="None",
        markersize=3,
    )

    # # plot moving average of sharpened bounds
    # sharp_bound_avg = np.convolve(
    #     sharp_iters[:n_iters_plot],
    #     np.ones(MOVING_AVG_WINDOW) / MOVING_AVG_WINDOW,
    #     mode="same",
    # )
    # ax.plot(
    #     MOVING_AVG_WINDOW + np.arange(len(sharp_bound_avg)),
    #     sharp_bound_avg,
    #     label=f"{shorthand} (MA)",
    #     linestyle="--",
    #     color="blue",
    # )

    # plot moving min of sharp bound
    moving_min_sharp = np.min(
        np.lib.stride_tricks.sliding_window_view(
            sharp_iters[:n_iters_plot], MOVING_AVG_WINDOW
        ),
        axis=1,
    )
    ax.plot(
        MOVING_AVG_WINDOW + np.arange(len(moving_min_sharp)),
        moving_min_sharp,
        label=f"Sharpened bound (MM)",
        linestyle="--",
        color=sharp_line[0].get_color(),
    )

    # plot moving min of sharp mixed bound
    moving_min_sharp_mixed = np.min(
        np.lib.stride_tricks.sliding_window_view(
            sharp_mixed_iters[:n_iters_plot], MOVING_AVG_WINDOW
        ),
        axis=1,
    )
    ax.plot(
        MOVING_AVG_WINDOW + np.arange(len(moving_min_sharp_mixed)),
        moving_min_sharp_mixed,
        label=f"Sharpened Mixed Bound (MM)",
        linestyle="--",
        color=sharp_mixed_line[0].get_color(),
    )

    # plot average of two moving mins
    moving_mins_avg = (moving_min_sharp + moving_min_sharp_mixed) / 2
    ax.plot(
        MOVING_AVG_WINDOW + np.arange(len(moving_mins_avg)),
        moving_mins_avg,
        label=f"Sharpened Bound (MM Avg)",
        linestyle="--",
        color="black",
    )

    # plot upper bound at convergence
    ax.axhline(
        mixed_sharpened_cg_iteration_bound(
            convergence_eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
        ),
        linestyle="--",
        color=ax.lines[-1].get_color(),
        linewidth=0.8,
    )

    # plot actual number of iterations
    ax.axhline(
        len(convergence_eigenvalues) - 1,
        linestyle="-",
        color=ax.lines[-1].get_color(),
        linewidth=0.8,
        label=f"{shorthand} (actual)",
    )

    # plot y=x
    ax.plot(
        np.arange(n_iters_plot),
        np.arange(n_iters_plot),
        linestyle="--",
        color="black",
        linewidth=0.8,
        label="y=x",
    )

    # log scale y-axis
    ax.set_yscale("log")

    # grid settings
    ax.grid(axis="x", which="both", linestyle="--", linewidth=0.7)
    ax.grid(axis="y", which="both", linestyle=":", linewidth=0.5)


def style_figure(fig, axs, shorthand, meshes, coef_funcs):
    # Add column titles (LaTeX, Nc as integer, no bold for compatibility)
    for col_idx, mesh_params in enumerate(meshes):
        H = mesh_params.coarse_mesh_size
        Nc = int(1 / H)
        ax = axs[0, col_idx] if hasattr(axs, "ndim") and axs.ndim == 2 else axs[0]
        ax.set_title(rf"$\mathbf{{H = 1/{Nc}}}$", fontsize=11)

    # Add row labels (rotated, bold, fontsize 9) at the beginning of each row
    for row_idx, coef_func in enumerate(coef_funcs):
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

    # figure title
    fig.suptitle(
        f"Sharpened CG Iteration Bound vs Iterations ({shorthand})",
        fontsize=16,
        fontweight="bold",
    )

    # tight layout for the figure
    fig.tight_layout(pad=1.3)


if ARGS.generate_output:
    for preconditioner in PRECONDITIONERS:
        fig, shorthand = plot_sharpened_bound_vs_iterations(
            preconditioner, meshes=PLOT_MESHES, show_bounds=SHOW_BOUNDS
        )
        fn = Path(__file__).name.replace("_fig.py", f"_{shorthand}")
        save_latex_figure(fn, fig)
if ARGS.show_output:
    figs = []
    for preconditioner in PRECONDITIONERS:
        fig, _ = plot_sharpened_bound_vs_iterations(
            preconditioner, meshes=PLOT_MESHES, show_bounds=SHOW_BOUNDS
        )
        figs.append(fig)
    plt.show()
