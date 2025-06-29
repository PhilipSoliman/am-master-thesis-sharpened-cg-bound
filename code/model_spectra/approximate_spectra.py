from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch

from lib import gpu_interface as gpu
from lib.boundary_conditions import HomogeneousDirichlet
from lib.eigenvalues import eigs
from lib.logger import LOGGER, PROGRESS
from lib.meshes import DefaultMeshParams, TwoLevelMesh
from lib.preconditioners import (
    AMSCoarseSpace,
    GDSWCoarseSpace,
    OneLevelSchwarzPreconditioner,
    Q1CoarseSpace,
    RGDSWCoarseSpace,
    TwoLevelSchwarzPreconditioner,
)
from lib.problem_type import ProblemType
from lib.problems import CoefFunc, DiffusionProblem, SourceFunc
from lib.solvers import CustomCG
from lib.utils import get_cli_args, save_latex_figure, set_mpl_cycler, set_mpl_style

# set logging level
LOGGER.setLevel("INFO")

# set matplotlib style & cycler
set_mpl_style()
set_mpl_cycler(colors=True)

# get cli args
ARGS = get_cli_args()
FIGWIDTH = 5
FIGHEIGHT = 4

# setup for a diffusion problem
MESHES = [DefaultMeshParams.Nc64]
PROBLEM_TYPE = ProblemType.DIFFUSION
BOUNDARY_CONDITIONS = HomogeneousDirichlet(PROBLEM_TYPE)
SOURCE_FUNC = SourceFunc.CONSTANT
COEF_FUNCS = [
    CoefFunc.CONSTANT,
    CoefFunc.DOUBLE_SLAB_EDGE_INCLUSIONS,
]

# solver tolerance
RTOL = 1e-8

# initialize progress bar
progress = PROGRESS.get_active_progress_bar()
main_task = progress.add_task(
    "Calculating spectra for coefficient functions", total=len(MESHES)
)

# initialize figure and axes
fig, axs = plt.subplots(
    len(COEF_FUNCS),
    len(MESHES),
    figsize=(FIGWIDTH * len(MESHES), FIGHEIGHT),
    squeeze=True,
    sharex=True,
    sharey=True,
)

# main loop over meshes, coefficients, and preconditioners
for i, mesh_params in enumerate(MESHES):
    two_mesh = TwoLevelMesh.load(mesh_params, progress=progress)
    axes = axs[:, i] if axs.ndim == 2 else axs

    # output directory
    output_dir_mesh = two_mesh.save_dir / "diffusion_spectra"
    for coef_func, ax in zip(COEF_FUNCS, axes):
        # output directory 
        output_dir_coef = output_dir_mesh / f"coef={coef_func.short_name}"
        output_dir_coef.mkdir(parents=True, exist_ok=True)

        # Create the diffusion problem instance
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

        # get preconditioners
        precond_task = progress.add_task("Getting preconditioners", total=6)
        preconditioners: list[
            None | OneLevelSchwarzPreconditioner | TwoLevelSchwarzPreconditioner
        ] = [
            # None, NOTE: original system can take a long time to solve, so we skip it here.
            # Also causes GPU memory issues due to large lanczos matrices. Even estimating condition number with scipy is troublesome.
            # This should be treated elsewhere.
            TwoLevelSchwarzPreconditioner(
                A,
                diffusion_problem.fes,
                diffusion_problem.two_mesh,
                coarse_space=GDSWCoarseSpace,
                progress=progress,
                coarse_only=True,
            ),
            TwoLevelSchwarzPreconditioner(
                A,
                diffusion_problem.fes,
                diffusion_problem.two_mesh,
                coarse_space=RGDSWCoarseSpace,
                progress=progress,
                coarse_only=True,
            ),
            # TwoLevelSchwarzPreconditioner( # NOTE: AMS is not robust, might be something wrong with the coarse space.
            #     A,
            #     diffusion_problem.fes,
            #     diffusion_problem.two_mesh,
            #     coarse_space=AMSCoarseSpace,
            #     progress=progress,
            #     coarse_only=True,
            # ),
        ]
        progress.remove_task(precond_task)

        # initialize CG solver
        custom_cg = CustomCG(
            A,
            b,
            u,
            tol=RTOL,
            progress=progress,
        )

        # get spectrum of (preconditioned) systems
        spectra_task = progress.add_task(
            "Computing spectra", total=len(preconditioners)
        )
        spectra = {}
        cond_numbers = []
        M1 = OneLevelSchwarzPreconditioner(A, diffusion_problem.fes, progress=progress).as_linear_operator()
        for i, preconditioner in enumerate(preconditioners):
            # get shorthand for preconditioner
            shorthand = preconditioner.short_name

            # output filename
            fn = output_dir_coef / f"{shorthand}.npy"

            # get preconditioner as linear operator
            M2 = preconditioner.as_linear_operator()
            M = sp.linalg.LinearOperator(
                A.shape, lambda x: M1.matvec(x) + M2.matvec(x)
            )

            # get eigenvalues from CG iterations
            LOGGER.info(f"Performing PCG iterations with {shorthand} preconditioner")
            _, success = custom_cg.sparse_solve(M, save_residuals=False)
            LOGGER.info("Computing approximate eigenvalues")
            eigenvalues = custom_cg.get_approximate_eigenvalues_gpu()

            # save eigenvalues to numpy array
            np.save(fn, eigenvalues)

            # save eigenvalues and condition numbers
            spectra[f"{shorthand:<10}"] = eigenvalues
            cond_numbers.append(
                np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))
                if len(eigenvalues) > 0 and np.min(np.abs(eigenvalues)) > 0
                else np.nan
            )
            progress.advance(spectra_task)
        progress.remove_task(spectra_task)

        # plot spectrum of A
        plot_task = progress.add_task(
            "Plotting spectra", total=len(spectra) + 1
        )  # + 1 for formatting
        for idx, (shorthand, eigenvalues) in enumerate(spectra.items()):
            ax.plot(
                np.real(eigenvalues),
                np.full_like(eigenvalues, idx),
                marker="x",
                linestyle="None",
            )
            progress.advance(plot_task)

        ax.set_yticks(range(len(spectra)), list(spectra.keys()))
        ax.set_xscale("log")
        ax.grid(axis="x")
        ax.grid()
        ax2 = ax.twinx()

        def format_cond(c):
            if np.isnan(c):
                return "n/a"
            mantissa, exp = f"{c:.1e}".split("e")
            exp = int(exp)
            return rf"${mantissa} \times 10^{{{exp}}}$"

        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(range(len(cond_numbers)))
        ax2.set_yticklabels(
            [format_cond(c) if not np.isnan(c) else "n/a" for c in cond_numbers]
        )
        progress.advance(plot_task)
        progress.remove_task(plot_task)

    # increment main task progress
    progress.advance(main_task)

# Add column titles (LaTeX, Nc as integer, no bold for compatibility)
for col_idx, mesh_params in enumerate(MESHES):
    H = mesh_params.coarse_mesh_size
    Nc = int(round(1 / H))
    ax = axs[0, col_idx] if axs.ndim == 2 else axs[0]
    ax.set_title(rf"$H = 1/{Nc}$", fontsize=11)

# Add row labels (rotated, bold, fontsize 9) at the beginning of each row
for row_idx, coef_func in enumerate(COEF_FUNCS):
    # Get the y-position as the center of the row of axes
    if axs.ndim == 2:
        ax = axs[row_idx, 0]
    else:
        ax = axs[row_idx]
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
    )

# tight layout for the figure
fig.tight_layout()

# stop progress bar
progress.soft_stop()

if ARGS.generate_output:
    fn = Path(__file__).name.replace("_fig.py", "")
    save_latex_figure(fn, fig)
if ARGS.show_output:
    plt.show()
