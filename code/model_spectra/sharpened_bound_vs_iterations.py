from pathlib import Path

import numpy as np
from approximate_spectra import (
    COEF_FUNCS,
    MESHES,
    PRECONDITIONERS,
    RTOL,
    get_spectrum_save_path,
)

from hcmsfem.eigenvalues import eigs
from hcmsfem.logger import LOGGER, PROGRESS
from hcmsfem.solvers import (
    CGIterationBound,
    CustomCG,
    multi_cluster_cg_iteration_bound,
    multi_tail_cluster_cg_iteration_bound,
)

# tolerance
LOG_RTOL = np.log(RTOL)

# max number of iterations to calculate (n_iters = min(N_ITERATIONS, len(alpha)-1))
N_ITERATIONS = 1200

# CG iteration bound update frequency
UPDATE_FREQUENCY = 5


def calculate_sharpened_bound_vs_iterations():
    progress = PROGRESS.get_active_progress_bar()
    main_task = progress.add_task(
        "Calculating upper bound vs iterations", total=len(MESHES)
    )
    main_desc = progress.get_description(main_task)
    main_desc += " ([bold]H = 1/{0:.0f}, CF = {1}[/bold], M = {2})"

    for mesh_params in MESHES:
        for coef_func in COEF_FUNCS:
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

                    # update main task description
                    progress.update(
                        main_task,
                        description=main_desc.format(
                            1 / mesh_params.coarse_mesh_size,
                            coef_func.short_name,
                            shorthand,
                        ),
                    )

                    # loop over CG iterations
                    num_iterations = min(N_ITERATIONS, len(alpha) - 1)
                    eigenvalue_task = progress.add_task(
                        f"Loop over CG iterations",
                        total=num_iterations,
                    )
                    eigenvalue_desc = (
                        progress.get_description(eigenvalue_task) + " ({})"
                    )
                    niters_multi_cluster = np.zeros(num_iterations, dtype=int)
                    niters_tail_cluster = np.full(num_iterations, np.nan, dtype=float)
                    cg_bound = CGIterationBound(
                        log_rtol=LOG_RTOL, exact_convergence=False
                    )
                    for j in range(num_iterations):
                        progress.update(
                            eigenvalue_task,
                            description=eigenvalue_desc.format(
                                "constructing lanczos matrix"
                            ),
                        )

                        lanczos_matrix = CustomCG.get_lanczos_matrix_from_coefficients(
                            alpha[: j + 1], beta[:j]
                        )

                        progress.update(
                            eigenvalue_task,
                            description=eigenvalue_desc.format(
                                "calculating eigenvalues"
                            ),
                        )
                        eigenvalues = eigs(lanczos_matrix)

                        progress.update(
                            eigenvalue_task,
                            description=eigenvalue_desc.format(
                                "applying sharpened bounds"
                            ),
                        )

                        # calculate sharpened bound
                        niter_multi_cluster = multi_cluster_cg_iteration_bound(
                            eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                        )
                        niters_multi_cluster[j] = niter_multi_cluster

                        # calculate sharpened mixed bound
                        niter_tail_cluster = multi_tail_cluster_cg_iteration_bound(
                            eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                        )
                        niters_tail_cluster[j] = niter_tail_cluster

                        # update CG iteration bound
                        if j % UPDATE_FREQUENCY == 0:
                            cg_bound.update(eigenvalues)

                        # update progress bar
                        progress.advance(eigenvalue_task)
                    progress.remove_task(eigenvalue_task)

                    # show the CG iteration bound
                    cg_bound.show()

                    # save arrays
                    np.savez(
                        fp,
                        alpha=array_zip["alpha"],
                        beta=array_zip["beta"],
                        niters_multi_cluster=niters_multi_cluster,
                        niters_tail_cluster=niters_tail_cluster,
                        eigenvalues=array_zip["eigenvalues"],
                        classic_bound=cg_bound.classic_l,
                        multi_cluster_bound=cg_bound.multi_cluster_l,
                        tail_cluster_bound=cg_bound.tail_cluster_l,
                        estimate=cg_bound.estimate_l,
                        cluster_convergence_iterations=cg_bound.iterations,
                    )

                else:
                    # Provide a clickable link to the script in the repo using Rich markup with absolute path
                    approx_path = Path(__file__).parent / "approximate_spectra.py"
                    LOGGER.error(
                        f"File %s does not exist. Run '[link=file:{approx_path}]approximate_spectra.py[/link]' first.",
                        fp,
                    )
                    exit()

        # advance main task
        progress.advance(main_task)

    # stop progress bar
    progress.soft_stop()


if __name__ == "__main__":
    calculate_sharpened_bound_vs_iterations()
