from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from lib.custom_cg import CustomCG
from lib.utils import get_cli_args, save_latex_figure

###################
# CONSTANT INPUTS #
###################
ARGS = get_cli_args()

# plot
FIGWIDTH = 5
LEGEND_HEIGHT = 0.1
LEVELS = 5
RESOLUTION = int(1e2)

# CG convergence
TOLERANCE = 1e-6

# spectrum
MIN_EIG = 1e-5  # reciprocal of the contrast of problem coefficient
RIGHT_CLUSTER_CONDITION_NUMBER = 1e2  # condition number bound for contrast=1

# cluster
LEFT_CLUSTER_MIN_WIDTH = 1e-10
LEFT_CLUSTER_MAX_WIDTH = 1e-6
MAX_SPECTRAL_WIDTH = 1e10

###########################
# PERFORMANCE CALCULATION #
###########################
# performance matrix
performance = np.zeros(
    (RESOLUTION + 1, RESOLUTION + 1)
)  # left cluster widths vs spectral widths

# cluster variables
left_cluster_width_multiplier = (LEFT_CLUSTER_MAX_WIDTH / LEFT_CLUSTER_MIN_WIDTH) ** (
    1 / RESOLUTION
)
left_cluster_widths = np.array(
    [
        LEFT_CLUSTER_MIN_WIDTH * left_cluster_width_multiplier**i
        for i in range(RESOLUTION + 1)
    ]
)
min_spectral_width = 0.5 + RIGHT_CLUSTER_CONDITION_NUMBER + (RIGHT_CLUSTER_CONDITION_NUMBER - 1) * MIN_EIG/LEFT_CLUSTER_MIN_WIDTH
spectral_width_multiplier = (MAX_SPECTRAL_WIDTH / min_spectral_width) ** (
    1 / RESOLUTION
)
spectral_widths = np.array(
    [min_spectral_width * spectral_width_multiplier**i for i in range(RESOLUTION + 1)]
)
b_1s = MIN_EIG + left_cluster_widths
c_1s = (MIN_EIG + b_1s) / 2


# iteration bounds
def compute_bound_for_width(i):
    p_i = np.zeros(len(spectral_widths))
    b_1 = b_1s[i]
    w_1 = left_cluster_widths[i]
    c_1 = c_1s[i]
    for j, d_ec in enumerate(spectral_widths):
        b_n = c_1 + d_ec * w_1
        a_n = b_n / RIGHT_CLUSTER_CONDITION_NUMBER
        clusters = [(MIN_EIG, b_1), (a_n, b_n)]

        # classical CG bound
        m_c = CustomCG.calculate_iteration_upperbound_static(
            b_n / MIN_EIG, np.log(TOLERANCE), exact_convergence=True
        )

        # improved CG bound
        m_s = CustomCG.calculate_improved_cg_iteration_upperbound_static(
            clusters, tol=TOLERANCE, exact_convergence=True
        )

        # performance ratio
        p_i[j] = m_c / m_s
    return i, p_i


with ThreadPoolExecutor() as executor:
    desc = "Calculating performance surface"
    futures = [
        executor.submit(compute_bound_for_width, i)
        for i in range(len(left_cluster_widths))
    ]
    with tqdm(total=len(futures), desc=desc) as pbar:
        for future in futures:
            i, p_i = future.result()
            performance[i, :] = p_i
            pbar.update(1)

#######################
# PERFORMANCE SURFACE #
#######################
fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGWIDTH))

# color map (not viridis)
cmap = plt.get_cmap("plasma")

# plot surface
log_p = np.log10(performance)
surf = ax.contourf(
    spectral_widths,
    left_cluster_widths,
    log_p,
    cmap=cmap,
    antialiased=True,
    levels=LEVELS,
)
ax.contour(
    spectral_widths,
    left_cluster_widths,
    log_p,
    colors="black",
    linewidths=1.0,
    linestyles="solid",
    levels=LEVELS,
)

# meta info
plt.colorbar(surf, ax=ax, label=r"$\log_{10}\left(\frac{m_c}{m_s}\right)$", pad=0.01)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"$d_{ec}$")
ax.set_ylabel(r"$w_1$")

fig.tight_layout()

if ARGS.generate_output:
    save_latex_figure("performance_spectral_width", fig)
if ARGS.show_output:
    plt.show()