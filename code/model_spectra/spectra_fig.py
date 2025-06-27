from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

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
from lib.utils import get_cli_args, save_latex_figure, set_mpl_cycler, set_mpl_style

# set logging level
LOGGER.setLevel("WARN")

# set matplotlib style & cycler
set_mpl_style()
set_mpl_cycler(colors=True)

# get cli args
ARGS = get_cli_args()
FIGWIDTH = 6
FIGHEIGHT = 4

# setup for a diffusion problem
PROBLEM_TYPE = ProblemType.DIFFUSION
BOUNDARY_CONDITIONS = HomogeneousDirichlet(PROBLEM_TYPE)
SOURCE_FUNC = SourceFunc.CONSTANT  # Source function
COEF_FUNCS = [
    CoefFunc.CONSTANT,
    CoefFunc.DOUBLE_SLAB_EDGE_INCLUSIONS,
]

# initialize progress bar
progress = PROGRESS.get_active_progress_bar()
main_task = progress.add_task(
    "Calculating spectra for coefficient functions", total=len(COEF_FUNCS)
)

# load mesh
TWO_MESH = TwoLevelMesh.load(DefaultMeshParams.Nc4, progress=progress)

# initialize figure and axes
fig, axs = plt.subplots(
    len(COEF_FUNCS), 1, figsize=(FIGWIDTH, FIGHEIGHT), squeeze=True, sharex=True
)

# main loop over coefficient functions & axes
for coef_func, ax in zip(COEF_FUNCS, axs):
    # Create the diffusion problem instance
    diffusion_problem = DiffusionProblem(
        boundary_conditions=BOUNDARY_CONDITIONS,
        mesh=TWO_MESH,
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
        None,
        OneLevelSchwarzPreconditioner(A, diffusion_problem.fes, progress=progress),
        TwoLevelSchwarzPreconditioner(
            A,
            diffusion_problem.fes,
            diffusion_problem.two_mesh,
            coarse_space=Q1CoarseSpace,
            progress=progress,
            coarse_only=True,
        ),
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
        TwoLevelSchwarzPreconditioner(
            A,
            diffusion_problem.fes,
            diffusion_problem.two_mesh,
            coarse_space=AMSCoarseSpace,
            progress=progress,
            coarse_only=True,
        ),
    ]
    progress.remove_task(precond_task)

    # get spectrum of (preconditioned) systems
    spectra_task = progress.add_task("Computing spectra", total=len(preconditioners))
    spectra = {}
    cond_numbers = np.zeros(len(preconditioners))
    M1 = sp.csc_matrix(A.shape, dtype=float)
    for i, preconditioner in enumerate(preconditioners):
        M = sp.eye(A.shape[0], dtype=float).tocsc()  # identity matrix
        if i == 1:  # 1-level schwarz preconditioner
            M1 = preconditioner.as_full_system()  # calculate M1 only once
            M = M1
        elif i > 1:  # 2-level schwarz preconditioners
            M2 = preconditioner.as_full_system()
            M = M1 + M2  # combine M1 and M2
        eigenvalues = eigs(M @ A)
        spectra[f"{getattr(preconditioner, 'short_name', 'None'):<10}"] = eigenvalues
        cond_numbers[len(spectra) - 1] = (
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

    # Set y-ticks and labels
    ax.set_yticks(range(len(spectra)), list(spectra.keys()))
    ax.set_xscale("log")
    ax.set_title(
        f"$\sigma({{M^{{-1}}}}A)$ for $\mathcal{{C}}$ = {coef_func.name}, $f=$ {SOURCE_FUNC.name}"
    )
    ax.grid(axis="x")
    ax.grid()
    ax2 = ax.twinx()

    # add condition numbers on right axis
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

    # main task progress update
    progress.advance(main_task)

# tight layout for the figure
fig.tight_layout()

# stop progress bar
progress.soft_stop()

if ARGS.generate_output:
    fn = Path(__file__).name.replace("_fig.py", "")
    save_latex_figure(fn, fig)
if ARGS.show_output:
    plt.show()
