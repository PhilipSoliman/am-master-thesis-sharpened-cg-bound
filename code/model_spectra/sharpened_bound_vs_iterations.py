from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from approximate_spectra import (
    COEF_FUNCS,
    MESHES,
    PRECONDITIONERS,
    get_spectrum_save_path,
)

from hcmsfem.cli import get_cli_args
from hcmsfem.logger import LOGGER
from hcmsfem.plot_utils import save_latex_figure, set_mpl_cycler, set_mpl_style
from hcmsfem.solvers import sharpened_cg_iteration_bound

# set matplotlib style & cycler
set_mpl_style()
set_mpl_cycler(colors=True)

# get cli args
ARGS = get_cli_args()
FIGWIDTH = 5
FIGHEIGHT = 2

# initialize figure and axes
fig, axs = plt.subplots(
    len(COEF_FUNCS),
    len(MESHES),
    figsize=(FIGWIDTH * len(MESHES), FIGHEIGHT * len(COEF_FUNCS)),
    squeeze=False,
    sharex=True,
    sharey=True,
)

# GOAL: get an idea of how fast the sharpened bound converges to its final prediction of the number of iterations
# TODO 1: rerun approximate_spectra.py to get alpha and beta cg arrays
# TODO 2: for every coefficient function, mesh size and preconditioner, calculate the eigenvalues
# TODO 3: feed eigenvalues to sharpened_cg_iteration_bound and plot the number of iterations it calculates
# TODO 4: plot a horizontal line for the number of iterations that the sharpened bound predicts for the eigenspectrum at convergence
