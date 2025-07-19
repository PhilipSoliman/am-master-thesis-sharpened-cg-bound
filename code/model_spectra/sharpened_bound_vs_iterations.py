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
from hcmsfem.logger import LOGGER, PROGRESS
from hcmsfem.plot_utils import save_latex_figure, set_mpl_cycler, set_mpl_style
from hcmsfem.solvers import (
    CustomCG,
    mixed_sharpened_cg_iteration_bound,
    sharpened_cg_iteration_bound,
)

# set matplotlib style & cycler
set_mpl_style()
set_mpl_cycler(colors=True)

# get cli args
ARGS = get_cli_args()
FIGWIDTH = 5
FIGHEIGHT = 2

# define preconditioners and coefficient functions
preconditioners = [PRECONDITIONERS[1]] 
coef_funcs = [COEF_FUNCS[2]]  # only vertex centered and edge slabs

# tolerance
LOG_RTOL = np.log(RTOL)

# number of iterations to calculate
N_ITERATIONS = 1000
MOVING_AVG_WINDOW = 10

# initialize figure and axes
fig, axs = plt.subplots(
    len(coef_funcs),
    len(MESHES),
    figsize=(FIGWIDTH * len(MESHES), FIGHEIGHT * len(coef_funcs)),
    squeeze=False,
    sharex=True,
    sharey=True,
)

# GOAL: get an idea of how fast the sharpened bound converges to its final prediction of the number of iterations
# TODO 1: rerun approximate_spectra.py to get alpha and beta cg arrays: DONE

# initialize progress bar
progress = PROGRESS.get_active_progress_bar()
main_task = progress.add_task(
    "Calculating upper bound vs iterations", total=len(MESHES)
)
main_desc = progress.get_description(main_task)
main_desc += " ([bold]H = 1/{0:.0f}, CF = {1}[/bold], M = {2})"

# main plot loop
for i, mesh_params in enumerate(MESHES):
    axes = axs[:, i]
    for coef_func, ax in zip(coef_funcs, axes):
        niters_sharp = {}
        niters_sharp_mixed = {}
        final_bounds = []
        for preconditioner_cls, coarse_space_cls in preconditioners:
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

                # update main task description
                progress.update(
                    main_task,
                    description=main_desc.format(
                        1 / mesh_params.coarse_mesh_size,
                        coef_func.short_name,
                        shorthand,
                    ),
                )

                # loop over iterations
                num_iterations = min(N_ITERATIONS, len(alpha) - 1)
                eigenvalue_task = progress.add_task(
                    f"Calculating upperbound",
                    total=num_iterations + 1,
                )
                eigenvalue_desc = progress.get_description(eigenvalue_task) + " ({})"
                niters_sharp[shorthand] = np.zeros(num_iterations, dtype=int)
                niters_sharp_mixed[shorthand] = np.full(
                    num_iterations, np.nan, dtype=float
                )
                for j in range(num_iterations + 1):
                    progress.update(
                        eigenvalue_task,
                        description=eigenvalue_desc.format(
                            "constructing lanczos matrix"
                        ),
                    )
                    if j < num_iterations:
                        lanczos_matrix = CustomCG.get_lanczos_matrix_from_coefficients(
                            alpha[: j + 1], beta[:j]
                        )
                    else:
                        # last iteration, use full alpha and beta
                        lanczos_matrix = CustomCG.get_lanczos_matrix_from_coefficients(
                            alpha, beta
                        )

                    progress.update(
                        eigenvalue_task,
                        description=eigenvalue_desc.format("calculating eigenvalues"),
                    )
                    eigenvalues = eigs(lanczos_matrix)

                    progress.update(
                        eigenvalue_task,
                        description=eigenvalue_desc.format(
                            "applying sharpened bound(s)"
                        ),
                    )
                    niter_sharp_mixed = mixed_sharpened_cg_iteration_bound(
                        eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                    )
                    if j < num_iterations:
                        # calculate sharpened bound
                        try:
                            niter_sharp = sharpened_cg_iteration_bound(
                                eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                            )
                            niters_sharp[shorthand][j] = niter_sharp
                        except ValueError as e:
                            LOGGER.info(
                                f"Skipping sharpened bound calculation for {shorthand} at iteration {j}: {e}"
                            )

                        # calculate sharpened mixed bound
                        niters_sharp_mixed[shorthand][j] = niter_sharp_mixed
                    else:
                        # last iteration, use full alpha and beta and sharpened mixed bound
                        final_bounds.append(niter_sharp_mixed)

                    # update progress bar
                    progress.advance(eigenvalue_task)
                progress.remove_task(eigenvalue_task)

            else:
                # Provide a clickable link to the script in the repo using Rich markup with absolute path
                approx_path = Path(__file__).parent / "approximate_spectra.py"
                LOGGER.error(
                    f"File %s does not exist. Run '[link=file:{approx_path}]approximate_spectra.py[/link]' first.",
                    fp,
                )
                exit()

        # TODO 3 (continued): ...and plot the number of iterations it calculates
        for idx, shorthand in enumerate(niters_sharp_mixed.keys()):
            # plot the two versions of the sharpened bound
            sharp_iters = niters_sharp[shorthand]
            ax.plot(
                range(len(sharp_iters)),
                sharp_iters,
                label=shorthand,
                marker="v",
                markersize=3,
            )

            sharp_mixed_iters = niters_sharp_mixed[shorthand]
            ax.plot(
                range(len(sharp_mixed_iters)),
                sharp_mixed_iters,
                label=shorthand,
                marker="^",
                markersize=3,
            )

            # TODO 4: plot a horizontal line for the number of iterations that the sharpened bound predicts for the eigenspectrum at convergence
            ax.axhline(
                final_bounds[idx],
                linestyle="--",
                color=ax.lines[-1].get_color(),
                linewidth=0.8,
            )

            # plot moving average of sharpened bounds
            sharp_bound_avg = np.convolve(
                sharp_iters,
                np.ones(MOVING_AVG_WINDOW) / MOVING_AVG_WINDOW,
                mode="valid",
            )
            ax.plot(
                MOVING_AVG_WINDOW + np.arange(len(sharp_bound_avg)),
                sharp_bound_avg,
                label=f"{shorthand} (MA)",
                linestyle="--",
                color="blue",
            )
            
            # plot moving min
            moving_min = np.min(np.lib.stride_tricks.sliding_window_view(sharp_iters, MOVING_AVG_WINDOW), axis=1)
            ax.plot(
                MOVING_AVG_WINDOW + np.arange(len(moving_min)),
                moving_min,
                label=f"{shorthand} (MM)",
                linestyle="--",
                color="green",
            )

        # plot y=x
        ax.plot(
            np.arange(N_ITERATIONS),
            np.arange(N_ITERATIONS),
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

    # advance main task
    progress.advance(main_task)

# Add column titles (LaTeX, Nc as integer, no bold for compatibility)
for col_idx, mesh_params in enumerate(MESHES):
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

# tight layout for the figure
fig.tight_layout(pad=1.3)

# stop progress bar
progress.soft_stop()

if ARGS.generate_output:
    fn = Path(__file__).name.replace("_fig.py", "")
    save_latex_figure(fn, fig)
if ARGS.show_output:
    plt.show()
