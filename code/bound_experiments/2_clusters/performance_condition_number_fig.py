from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from hcmsfem.cli import get_cli_args
from hcmsfem.plot_utils import save_latex_figure
from hcmsfem.solvers import multi_cluster_cg_iteration_bound

###################
# CONSTANT INPUTS #
###################
CLI_ARGS = get_cli_args()

# plot
FIGWIDTH = 5
LEGEND_HEIGHT = 0.1
LEVELS = 5
RESOLUTION = int(1e2)

# CG convergence
TOLERANCE = 1e-6

# spectrum
MAX_CONDITION_NUMBER = 1e10  # maximum global condition number
MIN_EIG = 1e-8  # reciprocal of the contrast of problem coefficient
RIGHT_CLUSTER_CONDITION_NUMBER = 1e2  # condition number bound for contrast=1

# cluster
LEFT_CLUSTER_MIN_WIDTH_FACTOR = 1e-2
LEFT_CLUSTER_MAX_WIDTH_FACTOR = 1e2

###########################
# PERFORMANCE CALCULATION #
###########################
# performance matrix
performance = np.zeros(
    (RESOLUTION + 1, RESOLUTION + 1)
)  # left cluster widths vs spectral widths

# cluster variables
left_cluster_min_width = MIN_EIG * LEFT_CLUSTER_MIN_WIDTH_FACTOR
left_cluster_max_width = MIN_EIG * LEFT_CLUSTER_MAX_WIDTH_FACTOR
left_cluster_width_multiplier = (left_cluster_max_width / left_cluster_min_width) ** (
    1 / RESOLUTION
)
left_cluster_widths = np.array(
    [
        left_cluster_min_width * left_cluster_width_multiplier**i
        for i in range(RESOLUTION + 1)
    ]
)
min_condition_number = RIGHT_CLUSTER_CONDITION_NUMBER * (
    1 + left_cluster_min_width / MIN_EIG
)
condition_number_multiplier = (MAX_CONDITION_NUMBER / min_condition_number) ** (
    1 / RESOLUTION
)
condition_numbers = np.array(
    [
        min_condition_number * condition_number_multiplier**i
        for i in range(RESOLUTION + 1)
    ]
)
b_1s = MIN_EIG + left_cluster_widths
b_n = MIN_EIG * condition_numbers
a_n = b_n / RIGHT_CLUSTER_CONDITION_NUMBER
left_clusters = [(MIN_EIG, b_1s[i]) for i in range(RESOLUTION + 1)]
right_clusters = [(a_n[i], b_n[i]) for i in range(RESOLUTION + 1)]

# classical CG bound
convergence_factors = (np.sqrt(condition_numbers) - 1) / (
    np.sqrt(condition_numbers) + 1
)
m_c = np.ceil(np.log(TOLERANCE / 2) / np.log(convergence_factors))


# improved CG bound
def compute_bound_for_width(i):
    lcluster = left_clusters[i]
    m_s_i = np.zeros(RESOLUTION + 1)  # performance for this left cluster
    for j, rcluster in enumerate(right_clusters):
        clusters = [lcluster, rcluster]
        if rcluster[0] < lcluster[1]:
            # if the right cluster overlaps with the left cluster, skip
            m_s_i[j] = np.nan
            continue
        m_s = multi_cluster_cg_iteration_bound(
            clusters, log_rtol=np.log(TOLERANCE), exact_convergence=True
        )
        m_s_i[j] = m_s
    return i, m_s_i


with ThreadPoolExecutor() as executor:
    desc = "Calculating performance surface"
    futures = [
        executor.submit(compute_bound_for_width, i)
        for i in range(len(left_cluster_widths))
    ]
    with tqdm(total=len(futures), desc=desc) as pbar:
        for future in futures:
            i, m_s_i = future.result()
            performance[i, :] = m_c / m_s_i
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
    condition_numbers,
    left_cluster_widths,
    log_p,
    cmap=cmap,
    antialiased=True,
    levels=LEVELS,
)
ax.contour(
    condition_numbers,
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
ax.set_xlabel(r"$\kappa$")
ax.set_ylabel(r"$w_1$")

fig.tight_layout()

if CLI_ARGS.generate_output:
    save_latex_figure("performance_condition_number", fig)
if CLI_ARGS.show_output:
    plt.show()
