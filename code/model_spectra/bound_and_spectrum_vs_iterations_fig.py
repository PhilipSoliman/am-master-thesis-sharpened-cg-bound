from itertools import cycle
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from approximate_spectra import COEF_FUNCS, RTOL, get_spectrum_save_path
from matplotlib.ticker import MultipleLocator
from sharpened_bound_vs_iterations_fig import (
    FIGHEIGHT,
    FIGWIDTH,
    plot_bounds,
    style_figure,
)

from hcmsfem.cli import get_cli_args
from hcmsfem.eigenvalues import eigs
from hcmsfem.logger import LOGGER, PROGRESS
from hcmsfem.meshes import DefaultQuadMeshParams
from hcmsfem.plot_utils import CustomColors, save_latex_figure
from hcmsfem.preconditioners import (
    AMSCoarseSpace,
    GDSWCoarseSpace,
    RGDSWCoarseSpace,
    TwoLevelSchwarzPreconditioner,
)
from hcmsfem.problems import CoefFunc
from hcmsfem.solvers import (
    CGIterationBound,
    CustomCG,
    multi_cluster_cg_iteration_bound,
    multi_tail_cluster_cg_iteration_bound,
    partition_eigenspectrum,
    partition_eigenspectrum_tails,
)

CLI_ARGS = get_cli_args()

# tolerance
LOG_RTOL = np.log(RTOL)

# number of iterations to calculate
N_ITERATIONS = 300  # for RGDSW left cluster stabilizes around 1200 iterations

# spectrum plot
SPECTRUM_PLOT_FREQ = 5
low_colour = "#b7b7b7"  # CustomColors.SOFTSKY.value # "#e2e4fb"
high_colour = "#808080"  # grey
cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_gradient", [low_colour, high_colour]
)
SPECTRA_COLORS_CYCLER = cycle(
    [
        mcolors.to_hex(cmap(i))
        for i in np.linspace(0, 1, N_ITERATIONS // (10 * SPECTRUM_PLOT_FREQ))
    ]
)

# preconditioner and coarse space class to plot
PRECONDITIONERS = [
    (TwoLevelSchwarzPreconditioner, GDSWCoarseSpace),
    (TwoLevelSchwarzPreconditioner, AMSCoarseSpace),
    (TwoLevelSchwarzPreconditioner, RGDSWCoarseSpace),
]

# meshes to plot
MESHES = [
    DefaultQuadMeshParams.Nc64,
]

# coef_funcs to plot
COEF_FUNCS = [CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS]

FIGHEIGHT = 3

# legend
LEGEND_SIZE = 1.5
FONTSIZE = 10

# plot
PADDING = dict(hspace=0.1, left=0.12, right=0.97, top=0.9, bottom=0)


def plot_bound_and_spectrum(PRECONDITIONER) -> plt.Figure:
    progress = PROGRESS.get_active_progress_bar()
    main_task = progress.add_task(
        "Calculating upper bound vs iterations", total=len(MESHES)
    )
    main_desc = progress.get_description(main_task)
    main_desc += " ([bold]H = 1/{0:.0f}, CF = {1}[/bold], M = {2})"

    # initialize figure and axes for legend
    fig = plt.figure(
        figsize=(6 * len(MESHES), FIGHEIGHT * len(COEF_FUNCS) + LEGEND_SIZE)
    )
    # Reduce legend row height for less blank space
    gs = gridspec.GridSpec(
        3,
        len(COEF_FUNCS),
        height_ratios=[
            FIGHEIGHT * len(COEF_FUNCS),
            FIGHEIGHT * len(COEF_FUNCS),
            LEGEND_SIZE,
        ],
    )
    axs = []
    for i in range(2 * len(COEF_FUNCS)):
        mesh_axs = []
        for j in range(len(MESHES)):
            mesh_axs.append(
                fig.add_subplot(gs[i, j], sharex=None if j == 0 else mesh_axs[0])
            )
        axs.append(mesh_axs)
    axs = np.array(axs)
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis("off")

    # Further tighten layout between subplots and minimize edge padding
    fig.subplots_adjust(**PADDING)

    # main plot loop
    for i, mesh_params in enumerate(MESHES):
        for j, coef_func in enumerate(COEF_FUNCS):
            fp = get_spectrum_save_path(
                mesh_params, coef_func, PRECONDITIONER[0], PRECONDITIONER[1]
            )
            spectra = []
            if fp.exists():
                # Load the alpha and beta arrays from the saved numpy file
                array_zip = np.load(fp)
                alpha = array_zip["alpha"]
                beta = array_zip["beta"]

                # get shorthand
                shorthand = (
                    f"{PRECONDITIONER[0].SHORT_NAME}-{PRECONDITIONER[1].SHORT_NAME}"
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
                    total=num_iterations,
                )
                eigenvalue_desc = progress.get_description(eigenvalue_task) + " ({})"
                niters_multi_cluster = np.zeros(num_iterations, dtype=int)
                niters_tail_cluster = np.full(num_iterations, np.nan, dtype=float)
                for k in range(num_iterations):
                    progress.update(
                        eigenvalue_task,
                        description=eigenvalue_desc.format(
                            "constructing lanczos matrix"
                        ),
                    )

                    lanczos_matrix = CustomCG.get_lanczos_matrix_from_coefficients(
                        alpha[: k + 1], beta[:k]
                    )

                    progress.update(
                        eigenvalue_task,
                        description=eigenvalue_desc.format("calculating eigenvalues"),
                    )
                    eigenvalues = eigs(lanczos_matrix)
                    spectra.append(eigenvalues)

                    progress.update(
                        eigenvalue_task,
                        description=eigenvalue_desc.format(
                            "applying sharpened bound(s)"
                        ),
                    )

                    # calculate sharpened bound
                    niter_multi_cluster = multi_cluster_cg_iteration_bound(
                        eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                    )
                    niters_multi_cluster[k] = niter_multi_cluster

                    # calculate sharpened mixed bound
                    niter_tail_cluster = multi_tail_cluster_cg_iteration_bound(
                        eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                    )
                    niters_tail_cluster[k] = niter_tail_cluster

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

            # plot bounds
            bound_ax = axs[2 * j, i]
            plot_bounds(
                bound_ax,
                shorthand,
                array_zip["eigenvalues"],
                niters_multi_cluster,
                niters_tail_cluster,
                show_bound_markers=False,
                n_iters=N_ITERATIONS,
            )
            bound_ax.xaxis.set_major_locator(MultipleLocator(N_ITERATIONS // 10))
            bound_ax.tick_params(axis="x", which="both", top=False, labelbottom=False)
            bound_ax.set_xlim(0, N_ITERATIONS)
            bound_ax.set_ylabel("Bound $m(\\sigma(T_i))$")

            # plot spectra
            cg_iter_bound = CGIterationBound(log_rtol=LOG_RTOL, exact_convergence=False)
            spectra_ax = axs[2 * j + 1, i]
            for iteration, spectrum in enumerate(spectra):
                if iteration % SPECTRUM_PLOT_FREQ != 0:
                    continue
                spectra_ax.plot(
                    np.full_like(spectrum, iteration),
                    spectrum,
                    linestyle="None",
                    marker="x",
                    color=next(SPECTRA_COLORS_CYCLER),
                )

                # plot cluster indices
                partition_indices = partition_eigenspectrum(spectrum)
                spectra_ax.plot(
                    np.full_like(partition_indices, iteration),
                    spectrum[partition_indices],
                    linestyle="None",
                    marker="_",
                    color=CustomColors.NAVY.value,
                    markersize=10,
                )

                # plot tail-cluster indices
                partition_indices_mixed = partition_eigenspectrum_tails(
                    spectrum, log_rtol=LOG_RTOL
                )
                spectra_ax.plot(
                    np.full_like(partition_indices_mixed, iteration),
                    spectrum[partition_indices_mixed],
                    linestyle="None",
                    marker="|",
                    color=CustomColors.GOLD.value,
                    markersize=10,
                )

                # update CG bound with the current spectrum
                cg_iter_bound.update(spectrum)

            # print CG iteration bound information
            LOGGER.info(
                f"CG Iteration Bound for {shorthand} on mesh H = 1/{1 / mesh_params.coarse_mesh_size:.0f} with coef_func {coef_func.short_name} converged after {len(array_zip['eigenvalues'])} iterations."
            )
            LOGGER.info(str(cg_iter_bound))

            # style the spectra axis
            spectra_ax.set_yscale("log")
            spectra_ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
            spectra_ax.set_xlim(0, N_ITERATIONS)
            spectra_ax.xaxis.set_major_locator(MultipleLocator(N_ITERATIONS // 10))
            spectra_ax.set_xlabel("Iteration $i$")
            spectra_ax.set_ylabel("Spectrum $\\sigma(T_i)$")

        # advance main task
        progress.advance(main_task)

    # style the figure
    style_figure(fig, axs, shorthand, MESHES, COEF_FUNCS)

    # add legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        fontsize=FONTSIZE,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
    )

    # stop progress bar
    progress.soft_stop()

    return fig


if __name__ == "__main__":
    figs = []
    for PRECONDITIONER in PRECONDITIONERS:
        fig = plot_bound_and_spectrum(PRECONDITIONER)
        shorthand = PRECONDITIONER[0].SHORT_NAME + "-" + PRECONDITIONER[1].SHORT_NAME
        if CLI_ARGS.generate_output:
            fn = Path(__file__).name.replace("_fig.py", f"_{shorthand}")
            save_latex_figure(fn, fig)
        figs.append(fig)
    if CLI_ARGS.show_output:
        plt.show()
