from pathlib import Path
from typing import Optional, Type

import numpy as np
import scipy.sparse as sp

from hcmsfem.boundary_conditions import HomogeneousDirichlet
from hcmsfem.logger import LOGGER, PROGRESS
from hcmsfem.meshes import DefaultQuadMeshParams, MeshParams, TwoLevelMesh
from hcmsfem.preconditioners import (
    AMSCoarseSpace,
    CoarseSpace,
    GDSWCoarseSpace,
    OneLevelSchwarzPreconditioner,
    RGDSWCoarseSpace,
    TwoLevelSchwarzPreconditioner,
)
from hcmsfem.problem_type import ProblemType
from hcmsfem.problems import CoefFunc, DiffusionProblem, SourceFunc
from hcmsfem.solvers import CustomCG

# setup for a diffusion problem
MESHES = DefaultQuadMeshParams
PROBLEM_TYPE = ProblemType.DIFFUSION
if __name__ == "__main__":
    BOUNDARY_CONDITIONS = HomogeneousDirichlet(PROBLEM_TYPE)
SOURCE_FUNC = SourceFunc.CONSTANT
COEF_FUNCS = [
    CoefFunc.CONSTANT,
    CoefFunc.THREE_LAYER_VERTEX_INCLUSIONS,
    CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS,
]

# solver tolerance
RTOL = 1e-8

# define list of preconditioner to get the spectra for
PRECONDITIONERS: list[tuple[Type[TwoLevelSchwarzPreconditioner], Type[CoarseSpace]]] = [
    # None, NOTE: original system can take a long time to solve, so we skip it here.
    # Also causes (GPU) memory issues due to large lanczos matrices. Even estimating condition number with scipy is troublesome.
    # This should be treated elsewhere.
    (TwoLevelSchwarzPreconditioner, GDSWCoarseSpace),
    (TwoLevelSchwarzPreconditioner, RGDSWCoarseSpace),
    (TwoLevelSchwarzPreconditioner, AMSCoarseSpace),
]


def get_spectrum_save_path(
    mesh_params: MeshParams,
    coef_func: CoefFunc,
    preconditioner_cls: Type[
        OneLevelSchwarzPreconditioner | TwoLevelSchwarzPreconditioner
    ],
    coarse_space_cls: Optional[Type[CoarseSpace]] = None,
) -> Path:
    save_dir = (
        TwoLevelMesh.get_save_dir(mesh_params)
        / "diffusion_spectra"
        / f"coef={coef_func.short_name}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    fn = f"{preconditioner_cls.SHORT_NAME}-{coarse_space_cls.SHORT_NAME if coarse_space_cls else ''}.npz"
    return save_dir / fn


def calculate_spectra() -> None:
    # initialize progress bar
    progress = PROGRESS.get_active_progress_bar()
    main_task = progress.add_task("Calculating spectra", total=len(MESHES))
    desc = progress.get_description(main_task)
    desc += " ([bold]H = 1/{0:.0f}, CF = {1}[/bold], M = {2})"

    # main loop over meshes, coefficients, and preconditioners
    for i, mesh_params in enumerate(MESHES):
        two_mesh = TwoLevelMesh.load(mesh_params, progress=progress)
        for coef_func in COEF_FUNCS:
            progress.update(
                main_task,
                description=desc.format(
                    1 / two_mesh.coarse_mesh_size,
                    coef_func.short_name,
                    "initializing problem...",
                ),
            )

            # create the diffusion problem instance
            diffusion_problem = DiffusionProblem(
                boundary_conditions=BOUNDARY_CONDITIONS,
                mesh=two_mesh,
                source_func=SOURCE_FUNC,
                coef_func=coef_func,
                progress=progress,
            )

            # get discrete problem
            A, u, b = diffusion_problem.restrict_system_to_free_dofs(
                *diffusion_problem.assemble()
            )

            # initialize CG solver
            custom_cg = CustomCG(
                A,
                b,
                u,
                tol=RTOL,
                progress=progress,
            )

            # NOTE: we cache the 1-lvl preconditioner to save time for the 2-lvl preconditioners.
            progress.update(
                main_task,
                description=desc.format(
                    1 / two_mesh.coarse_mesh_size,
                    coef_func.short_name,
                    "getting 1st level...",
                ),
            )
            M1 = OneLevelSchwarzPreconditioner(
                A, diffusion_problem.fes, progress=progress
            ).as_linear_operator()

            # get spectra
            spectra_task = progress.add_task(
                "Computing spectra", total=len(PRECONDITIONERS)
            )
            for preconditioner_cls, coarse_space_cls in PRECONDITIONERS:
                progress.update(
                    main_task,
                    description=desc.format(
                        1 / two_mesh.coarse_mesh_size,
                        coef_func.short_name,
                        "getting 2nd level...",
                    ),
                )
                # initialize preconditioner
                preconditioner = preconditioner_cls(
                    A,
                    diffusion_problem.fes,
                    two_mesh,
                    coarse_space=coarse_space_cls,
                    progress=progress,
                    coarse_only=True,
                )

                # get shorthand for preconditioner
                shorthand = preconditioner.SHORT_NAME

                # set description for the current mesh, coefficient function and preconditioner
                progress.update(
                    main_task,
                    description=desc.format(
                        1 / two_mesh.coarse_mesh_size, coef_func.short_name, shorthand
                    ),
                )

                # get preconditioner as linear operator
                M2 = preconditioner.as_linear_operator()
                M = sp.linalg.LinearOperator(
                    A.shape, lambda x: M1.matvec(x) + M2.matvec(x)
                )

                # get eigenvalues from CG iterations
                LOGGER.info(
                    f"Performing PCG iterations with {shorthand} preconditioner"
                )
                _, _ = custom_cg.sparse_solve(M, save_residuals=False)
                LOGGER.info("Computing approximate eigenvalues")
                eigenvalues = custom_cg.get_approximate_eigenvalues_gpu()

                # get alpha and beta arrays
                alpha = custom_cg.alpha
                beta = custom_cg.beta

                # get save directory for the preconditioner's spectrum
                save_dir = get_spectrum_save_path(
                    mesh_params, coef_func, preconditioner_cls, coarse_space_cls
                )

                # save spectrum and related arrays to a single .npz file
                np.savez(
                    save_dir,
                    eigenvalues=eigenvalues,
                    alpha=alpha,
                    beta=beta,
                )

                progress.advance(spectra_task)
            progress.remove_task(spectra_task)

        # increment main task progress
        progress.advance(main_task)

    # stop progress bar
    progress.soft_stop()


if __name__ == "__main__":
    LOGGER.generate_log_file()
    calculate_spectra()
