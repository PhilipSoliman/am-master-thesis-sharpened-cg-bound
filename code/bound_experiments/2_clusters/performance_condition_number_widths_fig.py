from concurrent.futures import ThreadPoolExecutor

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm

from lib.solvers import CustomCG
from lib.utils import (
    CUSTOM_COLORS_SIMPLE,
    get_cli_args,
    save_latex_figure,
    set_mpl_cycler,
    set_mpl_style,
)

set_mpl_style()
set_mpl_cycler(colors=True, lines=True)

###################
# CONSTANT INPUTS #
###################
ARGS = get_cli_args()

# plot
FIGWIDTH = 6
LEGEND_HEIGHT = 0.1
LEVELS = 5
RESOLUTION = int(1e2)

# CG convergence
TOLERANCE = 1e-6

# spectrum
MIN_EIG = 1e-8  # reciprocal of the contrast of problem coefficient
MAX_CONDITION_NUMBER = 1e10  # maximum global condition number
RIGHT_CLUSTER_CONDITION_NUMBER = 1e1  # condition number bound for contrast=1
LEFT_CLUSTER_WIDTHS = np.array([1e-10, 1e-9, 1e-8, 1e-7, 1e-6])
min_condition_number = RIGHT_CLUSTER_CONDITION_NUMBER * (
    1 + LEFT_CLUSTER_WIDTHS[0] / MIN_EIG
)  # minimum global condition number


###########################
# PERFORMANCE CALCULATION #
###########################
# performance matrix
performance = np.zeros((len(LEFT_CLUSTER_WIDTHS), RESOLUTION + 1))
uniform_performance = np.zeros((RESOLUTION + 1,))

# cluster variables
condition_number_multiplier = (MAX_CONDITION_NUMBER / min_condition_number) ** (
    1 / RESOLUTION
)
condition_numbers = np.array(
    [
        min_condition_number * condition_number_multiplier**i
        for i in range(RESOLUTION + 1)
    ]
)
b_1s = MIN_EIG + LEFT_CLUSTER_WIDTHS
b_n = MIN_EIG * condition_numbers
a_n = b_n / RIGHT_CLUSTER_CONDITION_NUMBER
left_clusters = [(MIN_EIG, b_1) for b_1 in b_1s]
right_clusters = [(a_n[i], b_n[i]) for i in range(RESOLUTION + 1)]

# classical CG bound
convergence_factors = (np.sqrt(condition_numbers) - 1) / (
    np.sqrt(condition_numbers) + 1
)
m_c = np.ceil(np.log(TOLERANCE / 2) / np.log(convergence_factors))


# improved CG bound (per width)
def compute_bound_for_width(i):
    lcluster = left_clusters[i]
    m_s_i = np.zeros(RESOLUTION + 1)  # performance for this left cluster
    for j, rcluster in enumerate(right_clusters):
        clusters = [lcluster, rcluster]
        if rcluster[0] < lcluster[1]:
            # if the right cluster overlaps with the left cluster, skip
            m_s_i[j] = np.nan
            continue
        m_s = CustomCG.calculate_improved_cg_iteration_upperbound_static(
            clusters, tol=TOLERANCE, exact_convergence=True
        )
        m_s_i[j] = m_s
    return i, m_s_i


# improved CG bound (uniform performance)
for i, rcluster in enumerate(right_clusters):
    lcluster = (
        MIN_EIG,
        rcluster[0],
    )  # left cluster is almost touching the right cluster
    clusters = [lcluster, rcluster]
    m_s = CustomCG.calculate_improved_cg_iteration_upperbound_static(
        clusters, tol=TOLERANCE, exact_convergence=True
    )
    uniform_performance[i] = m_c[i] / m_s

with ThreadPoolExecutor() as executor:
    desc = "Calculating performance surface"
    futures = [
        executor.submit(compute_bound_for_width, i)
        for i in range(len(LEFT_CLUSTER_WIDTHS))
    ]
    with tqdm(total=len(futures), desc=desc) as pbar:
        for future in futures:
            i, m_s_i = future.result()
            performance[i, :] = m_c / m_s_i
            pbar.update(1)

######################
# PERFORMANCE CURVES #
######################
fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGWIDTH))

# plot performance curves per width
for i in range(len(LEFT_CLUSTER_WIDTHS)):
    ax.plot(
        condition_numbers,
        performance[i, :],
        label="$w_1 = 10^{" + f"{np.log10(LEFT_CLUSTER_WIDTHS[i]):.0f}" + "}$",
        lw=2,
    )


# plot uniform performance
ax.plot(
    condition_numbers,
    uniform_performance,
    label="Uniform spectrum",
    lw=2,
    linestyle="--",
    color="black",
)

# Define no improvement region
performance_below_1 = ShapelyPolygon(
    [
        [min_condition_number, 0],
        [MAX_CONDITION_NUMBER, 0],
        [MAX_CONDITION_NUMBER, 1],
        [min_condition_number, 1],
    ],
)
uniform_spectrum = ShapelyPolygon(
    [
        [min_condition_number, 1],
        *[[cond, perf] for cond, perf in zip(condition_numbers, uniform_performance)],
        [MAX_CONDITION_NUMBER, 1],
    ],
)
no_improvement_region = performance_below_1.intersection(uniform_spectrum)

if not no_improvement_region.is_empty:
    if no_improvement_region.geom_type == "Polygon":
        x, y = no_improvement_region.exterior.xy  # type: ignore
        ax.fill(x, y, color=CUSTOM_COLORS_SIMPLE[0], alpha=0.2, label="No improvement")
    elif no_improvement_region.geom_type == "MultiPolygon":
        for poly in no_improvement_region.geoms:  # type: ignore
            x, y = poly.exterior.xy
            ax.fill(
                x, y, color=CUSTOM_COLORS_SIMPLE[0], alpha=0.2, label="No improvement"
            )
else:
    print("No improvement region is empty, nothing to fill.")


ax.set_yscale("log")
ax.set_ylim((1e-1, 1e4))
ax.set_xscale("log")
ax.set_xlabel("Condition number $\\kappa$")
ax.set_ylabel("Performance $m_c / m_s$")

# meta info
ax.legend()
fig.tight_layout()

if ARGS.generate_output:
    save_latex_figure("performance_condition_number_widths", fig)
if ARGS.show_output:
    plt.show()
