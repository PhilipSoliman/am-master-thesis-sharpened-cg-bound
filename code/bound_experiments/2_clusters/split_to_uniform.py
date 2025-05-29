import matplotlib.pyplot as plt
import numpy as np

# from clibs.custom_cg import CustomCG
from lib.custom_cg import CustomCG
from lib.utils import (
    get_cli_args,
    mpl_graph_plot_style,
    save_latex_figure,
    set_mpl_cycler,
    set_mpl_style,
)

# CONSTANT INPUTS
ARGS = get_cli_args()
FIGWIDTH = 5
LEGEND_HEIGHT = 0.1
DOMAIN = (0, 20)
CODOMAIN = (-1, 1)
ZOOMFIT = True
MAX_EIG = 1.0
CONDITION_NUMBER = 1e8

# cluster parameters
BASE = 2
num_widths = int(np.ceil(np.log(CONDITION_NUMBER)/np.log(BASE)))
print(f"Condition number: {CONDITION_NUMBER}")
print(f"Number of clusters: {num_widths}")

# cluster locations
a_1 = MAX_EIG / CONDITION_NUMBER
b_1s = [a_1 * BASE**i for i in range(1,num_widths)]
a_n = b_1s[-1]
b_n	 = MAX_EIG
print(f"a_1: {a_1}")
print(f"b_1: {b_1s}")
print(f"a_n: {a_n}")
print(f"b_n: {b_n}")

# CG convergence parameters
TOLERANCE = 1e-6

# set matplotlib style
set_mpl_style()
set_mpl_cycler(lines=False, markers=True, colors=True)

# classical iteration bound
m_c = CustomCG.calculate_iteration_upperbound_static(CONDITION_NUMBER,np.log(TOLERANCE), exact_convergence=True)
print(f"Classical iteration bound: {m_c}")

# Improved iteration bound
for i, b_1 in enumerate(b_1s):
    clusters = [(a_1, b_1), (a_n, b_n)]
    m_i = CustomCG.calculate_improved_cg_iteration_upperbound_static(clusters, tol=TOLERANCE, exact_convergence=True)
    performance = m_c / m_i
    print(f"Improved iteration bound for cluster {i+1}: {m_i}, Performance: {performance:.2e}")
