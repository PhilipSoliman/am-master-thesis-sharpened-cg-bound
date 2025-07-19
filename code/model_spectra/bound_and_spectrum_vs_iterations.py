from pathlib import Path

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

from hcmsfem.eigenvalues import eigs
from hcmsfem.logger import LOGGER, PROGRESS
from hcmsfem.meshes import DefaultQuadMeshParams
from hcmsfem.preconditioners import (
    AMSCoarseSpace,
    GDSWCoarseSpace,
    RGDSWCoarseSpace,
    TwoLevelSchwarzPreconditioner,
)
from hcmsfem.problems import CoefFunc
from hcmsfem.solvers import (
    CustomCG,
    mixed_sharpened_cg_iteration_bound,
    partition_eigenspectrum,
    sharpened_cg_iteration_bound,
)

# tolerance
LOG_RTOL = np.log(RTOL)

# number of iterations to calculate
N_ITERATIONS = 300

# spectrum plot frequency
SPECTRUM_PLOT_FREQ = 5

# preconditioner and coarse space class to plot
PRECONDITIONER = (TwoLevelSchwarzPreconditioner, RGDSWCoarseSpace)

# meshes to plot
MESHES = [DefaultQuadMeshParams.Nc64]

# coef_funcs to plot
COEF_FUNCS = [CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS]

progress = PROGRESS.get_active_progress_bar()
main_task = progress.add_task(
    "Calculating upper bound vs iterations", total=len(MESHES)
)
main_desc = progress.get_description(main_task)
main_desc += " ([bold]H = 1/{0:.0f}, CF = {1}[/bold], M = {2})"


# initialize figure and axes
fig, axs = plt.subplots(
    len(COEF_FUNCS),
    2 * len(MESHES),
    figsize=(FIGWIDTH * 2 * len(MESHES), 2 * FIGHEIGHT * len(COEF_FUNCS)),
    squeeze=False,
    # sharey=True,
)

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
            shorthand = f"{PRECONDITIONER[0].SHORT_NAME}-{PRECONDITIONER[1].SHORT_NAME}"

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
            niters_sharp = np.zeros(num_iterations, dtype=int)
            niters_sharp_mixed = np.full(num_iterations, np.nan, dtype=float)
            for k in range(num_iterations):
                progress.update(
                    eigenvalue_task,
                    description=eigenvalue_desc.format("constructing lanczos matrix"),
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
                    description=eigenvalue_desc.format("applying sharpened bound(s)"),
                )

                # calculate sharpened bound
                niter_sharp = sharpened_cg_iteration_bound(
                    eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                )
                niters_sharp[k] = niter_sharp

                # calculate sharpened mixed bound
                niter_sharp_mixed = mixed_sharpened_cg_iteration_bound(
                    eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                )
                niters_sharp_mixed[k] = niter_sharp_mixed

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
        plot_bounds(
            axs[j, 2 * i],
            shorthand,
            array_zip["eigenvalues"],
            niters_sharp,
            niters_sharp_mixed,
            show_bounds=True,
            n_iters=N_ITERATIONS,
        )
        axs[j, 2 * i].xaxis.set_major_locator(MultipleLocator(SPECTRUM_PLOT_FREQ))

        # plot spectra
        for iteration, spectrum in enumerate(spectra):
            if iteration % SPECTRUM_PLOT_FREQ != 0:
                continue
            ax = axs[j, 2 * i + 1]
            ax.plot(
                np.full_like(spectrum, iteration),
                spectrum,
                linestyle="None",
                marker="x",
            )

            # ax.plot(
            #     iteration,
            #     spectrum[0],
            #     linestyle="None",
            #     marker="_",
            #     color="red",
            #     markersize=10,
            # )

            # plot partition indices
            partition_indices = partition_eigenspectrum(spectrum)
            ax.plot(
                np.full_like(partition_indices, iteration),
                spectrum[partition_indices],
                linestyle="None",
                marker="_",
                color="red",
                markersize=10,
            )
        ax.set_yscale("log")
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
        ax.xaxis.set_major_locator(MultipleLocator(SPECTRUM_PLOT_FREQ))

    # advance main task
    progress.advance(main_task)

# style the figure
style_figure(fig, axs, shorthand, MESHES, COEF_FUNCS)

# stop progress bar
progress.soft_stop()

plt.show()
