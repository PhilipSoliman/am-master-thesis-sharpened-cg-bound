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
    CustomCG,
    mixed_sharpened_cg_iteration_bound,
    sharpened_cg_iteration_bound,
)

# tolerance
LOG_RTOL = np.log(RTOL)

# max number of iterations to calculate (n_iters = min(N_ITERATIONS, len(alpha)-1))
N_ITERATIONS = 1000

# initialize progress bar
progress = PROGRESS.get_active_progress_bar()
main_task = progress.add_task(
    "Calculating upper bound vs iterations", total=len(MESHES)
)
main_desc = progress.get_description(main_task)
main_desc += " ([bold]H = 1/{0:.0f}, CF = {1}[/bold], M = {2})"

# main plot loop
for i, mesh_params in enumerate(MESHES):
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

                # loop over iterations
                num_iterations = min(N_ITERATIONS, len(alpha) - 1)
                eigenvalue_task = progress.add_task(
                    f"Calculating upperbound",
                    total=num_iterations,
                )
                eigenvalue_desc = progress.get_description(eigenvalue_task) + " ({})"
                niters_sharp = np.zeros(num_iterations, dtype=int)
                niters_sharp_mixed = np.full(num_iterations, np.nan, dtype=float)
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
                        description=eigenvalue_desc.format("calculating eigenvalues"),
                    )
                    eigenvalues = eigs(lanczos_matrix)

                    progress.update(
                        eigenvalue_task,
                        description=eigenvalue_desc.format(
                            "applying sharpened bound(s)"
                        ),
                    )

                    # calculate sharpened bound
                    niter_sharp = sharpened_cg_iteration_bound(
                        eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                    )
                    niters_sharp[j] = niter_sharp

                    # calculate sharpened mixed bound
                    niter_sharp_mixed = mixed_sharpened_cg_iteration_bound(
                        eigenvalues, log_rtol=LOG_RTOL, exact_convergence=False
                    )
                    niters_sharp_mixed[j] = niter_sharp_mixed

                    # update progress bar
                    progress.advance(eigenvalue_task)
                progress.remove_task(eigenvalue_task)

                # save arrays
                np.savez(
                    fp,
                    alpha=array_zip["alpha"],
                    beta=array_zip["beta"],
                    niters_sharp=niters_sharp,
                    niters_sharp_mixed=niters_sharp_mixed,
                    eigenvalues=array_zip["eigenvalues"],
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
