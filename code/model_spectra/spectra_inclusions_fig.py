from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spl
from tqdm import tqdm

from lib.boundary_conditions import HomogeneousDirichlet
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

# Set matplotlib style & cycler
set_mpl_style()
set_mpl_cycler(colors=True)

# get cli args
ARGS = get_cli_args()
FIGWIDTH = 6
FIGHEIGHT = 2

# setup for a diffusion problem
problem_type = ProblemType.DIFFUSION
boundary_conditions = HomogeneousDirichlet(problem_type)
lx, ly = 1.0, 1.0  # Length of the domain in x and y directions
coarrse_mesh_size = 0.3  # Size of the coarse mesh
refinement_levels = 2  # Number of times to refine the mesh
layers = 2  # Number of overlapping layers in the Schwarz Domain Decomposition
source_func = SourceFunc.CONSTANT  # Source function
coef_func = CoefFunc.INCLUSIONS  # Coefficient function

# Create the diffusion problem instance
diffusion_problem = DiffusionProblem(
    boundary_conditions=boundary_conditions,
    lx=lx,
    ly=ly,
    coarse_mesh_size=coarrse_mesh_size,
    refinement_levels=refinement_levels,
    layers=layers,
    source_func=source_func,
    coef_func=coef_func,
)

# get discrete problem
A, u, b = diffusion_problem.get_homogenized_system(*diffusion_problem.assemble())
print("Assembled discrete problem: A has shape ", A.shape)
n = A.shape[0]

# get preconditioners
preconditioners = {
    "None": None,
    "1-OAS": OneLevelSchwarzPreconditioner(A, diffusion_problem.fes),
    "2-Q1": TwoLevelSchwarzPreconditioner(
        A, diffusion_problem.fes, diffusion_problem.two_mesh, coarse_space=Q1CoarseSpace
    ),
    "2-GDSW": TwoLevelSchwarzPreconditioner(
        A,
        diffusion_problem.fes,
        diffusion_problem.two_mesh,
        coarse_space=GDSWCoarseSpace,
    ),
    "2-RGDSW": TwoLevelSchwarzPreconditioner(
        A,
        diffusion_problem.fes,
        diffusion_problem.two_mesh,
        coarse_space=RGDSWCoarseSpace,
    ),
    "2-AMS": TwoLevelSchwarzPreconditioner(
        A,
        diffusion_problem.fes,
        diffusion_problem.two_mesh,
        coarse_space=AMSCoarseSpace,
    ),
}

# get spectrum of (preconditioned) systems
spectra = {}
cond_numbers = np.zeros(len(preconditioners))
for shorthand, preconditioner in tqdm(
    preconditioners.items(), desc="Computing spectra"
):
    eigenvalues = spl.eigsh(
        preconditioner.as_linear_operator() if preconditioner is not None else A,
        k=n - 2,
        which="BE",  # gets eigenvalues on both ends of the spectrum
        return_eigenvectors=False,
    )
    spectra[shorthand] = eigenvalues
    cond_numbers[len(spectra) - 1] = (
        np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))
        if len(eigenvalues) > 0 and np.min(np.abs(eigenvalues)) > 0
        else np.nan
    )

# plot spectrum of A
fig = plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))

# Plot each spectrum at a different y
for idx, (shorthand, eigenvalues) in enumerate(spectra.items()):
    plt.plot(
        np.real(eigenvalues),
        np.full_like(eigenvalues, idx),
        marker="x",
        linestyle="None",
    )

# Set y-ticks and labels
plt.yticks(range(len(spectra)), list(spectra.keys()))
plt.xscale("log")
plt.title(
    f"$\sigma({{M^{{-1}}}}A)$ for $\mathcal{{C}}$ = {coef_func.name}, $f=$ {source_func.name}"
)
plt.grid(axis="x")
plt.grid()
ax = plt.gca()
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

plt.tight_layout()

if ARGS.generate_output:
    fn = Path(__file__).name.replace("_fig.py", "")
    save_latex_figure(fn, fig)
if ARGS.show_output:
    plt.show()
