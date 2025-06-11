from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
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

# CG convergence
TOLERANCE = 1e-6

# spectrum
MIN_EIGS = [1e-8, 1e2]  # reciprocal of the contrast of problem coefficient
LEFT_CLUSTER_WIDTHS_MULTIPLIERS = [1e-2, 1, 1e2, 1e4, 1e6]
RIGHT_CLUSTER_CONDITION_NUMBERS = [1e1, 1e2]  # condition number bound for contrast=1
MAX_CONDITION_NUMBER = 1e10  # maximum global condition number

# plot
FIGWIDTH = 8
FIGHEIGHT = (FIGWIDTH / len(RIGHT_CLUSTER_CONDITION_NUMBERS)) * len(MIN_EIGS)
LEGEND_HEIGHT = 0.1
RESOLUTION = int(1e2)
YLIMIT = (1e-1, 1e4)  # y-axis limits for performance plot

#############
# FUNCTIONS #
#############
# generating spectra
def get_spectra(right_cluster_condition_number: float, min_eig: float):
    
    LEFT_CLUSTER_WIDTHS = np.array(
        [min_eig * mult for mult in LEFT_CLUSTER_WIDTHS_MULTIPLIERS]
    )
    min_condition_number = right_cluster_condition_number * (
        1 + LEFT_CLUSTER_WIDTHS[0] / min_eig
    )  # minimum global condition number

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
    b_1s = min_eig + LEFT_CLUSTER_WIDTHS
    b_n = min_eig * condition_numbers
    a_n = b_n / right_cluster_condition_number
    left_clusters = [(min_eig, b_1) for b_1 in b_1s]
    right_clusters = [(a_n[i], b_n[i]) for i in range(RESOLUTION + 1)]
    return left_clusters, right_clusters, condition_numbers


# classical CG bound
def classical_bound(condition_numbers: np.ndarray) -> np.ndarray:
    convergence_factors = (np.sqrt(condition_numbers) - 1) / (
        np.sqrt(condition_numbers) + 1
    )
    return np.ceil(np.log(TOLERANCE / 2) / np.log(convergence_factors))


# improved CG bound
def compute_bound_for_width(i, left_cluster_i, right_clusters):
    lcluster = left_cluster_i
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
def compute_uniform_performance(
    right_clusters: list[tuple[float, float]], min_eig: float, m_c: np.ndarray
) -> np.ndarray:
    uniform_performance = np.zeros((RESOLUTION + 1,))
    for i, rcluster in enumerate(right_clusters):
        lcluster = (
            min_eig,
            rcluster[0],
        )  # left cluster is almost touching the right cluster
        clusters = [lcluster, rcluster]
        m_s = CustomCG.calculate_improved_cg_iteration_upperbound_static(
            clusters, tol=TOLERANCE, exact_convergence=True
        )
        uniform_performance[i] = m_c[i] / m_s

    return uniform_performance


# performance calculation
def calculate_performance(
    left_clusters: list[tuple[float, float]],
    right_clusters: list[tuple[float, float]],
    m_c: np.ndarray,
) -> np.ndarray:
    performance = np.zeros((len(LEFT_CLUSTER_WIDTHS_MULTIPLIERS), RESOLUTION + 1))
    with ThreadPoolExecutor() as executor:
        desc = "Calculating performance surface"
        futures = [
            executor.submit(
                compute_bound_for_width, i, left_clusters[i], right_clusters
            )
            for i in range(len(LEFT_CLUSTER_WIDTHS_MULTIPLIERS))
        ]
        with tqdm(total=len(futures), desc=desc) as pbar:
            for future in futures:
                i, m_s_i = future.result()
                performance[i, :] = m_c / m_s_i
            pbar.update(1)

    return performance


# plot performance curves per width
def plot_performance_curves(
    ax: Axes,
    condition_numbers: np.ndarray,
    performance: np.ndarray,
    uniform_performance: np.ndarray,
):
    for i, mult in enumerate(LEFT_CLUSTER_WIDTHS_MULTIPLIERS):
        ax.plot(
            condition_numbers,
            performance[i, :],
            lw=2,
        )
        # Annotate at the right end of each curve
        x_annot = condition_numbers[-1] * 10
        y_annot = performance[i, -1]
        ax.text(
            x_annot,
            y_annot,
            f"$w_1=10^{{{int(np.log10(min_eig * mult))}}}$",
            fontsize=8,
            va="center",
            ha="center",
            color=ax.lines[-1].get_color(),  # match curve color
            clip_on=False,
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

    performance_below_1 = ShapelyPolygon(
        [
            [condition_numbers[0], 0],
            [MAX_CONDITION_NUMBER, 0],
            [MAX_CONDITION_NUMBER, 1],
            [condition_numbers[0], 1],
        ],
    )
    uniform_spectrum = ShapelyPolygon(
        [
            [condition_numbers[0], 1],
            *[
                [cond, perf]
                for cond, perf in zip(condition_numbers, uniform_performance)
            ],
            [MAX_CONDITION_NUMBER, 1],
        ],
    )
    no_improvement_region = performance_below_1.intersection(uniform_spectrum)

    if not no_improvement_region.is_empty:
        if no_improvement_region.geom_type == "Polygon":
            x, y = no_improvement_region.exterior.xy  # type: ignore
            ax.fill(
                x, y, color=CUSTOM_COLORS_SIMPLE[0], alpha=0.2, label="No improvement"
            )
        elif no_improvement_region.geom_type == "MultiPolygon":
            for poly in no_improvement_region.geoms:  # type: ignore
                x, y = poly.exterior.xy
                ax.fill(
                    x,
                    y,
                    color=CUSTOM_COLORS_SIMPLE[0],
                    alpha=0.2,
                    label="No improvement",
                )
    else:
        print("No improvement region is empty, nothing to fill.")

    ax.set_yscale("log")
    ax.set_ylim(YLIMIT)


#########################
# CALCULATE PERFORMANCE #
#########################
fig, axs = plt.subplots(
    nrows=len(MIN_EIGS),
    ncols=len(RIGHT_CLUSTER_CONDITION_NUMBERS),
    figsize=(FIGWIDTH, FIGHEIGHT),
    sharex=True,
    sharey=True,
    squeeze=False,
)

for row, min_eig in enumerate(MIN_EIGS):
    for col, right_cluster_condition_number in enumerate(
        RIGHT_CLUSTER_CONDITION_NUMBERS
    ):
        ax = axs[row, col]
        left_clusters, right_clusters, condition_numbers = get_spectra(
            right_cluster_condition_number, min_eig
        )
        m_c = classical_bound(condition_numbers)
        performance = calculate_performance(left_clusters, right_clusters, m_c)
        uniform_performance = compute_uniform_performance(right_clusters, min_eig, m_c)
        plot_performance_curves(ax, condition_numbers, performance, uniform_performance)
        if row == 0 and col == 0:
            ax.legend(fontsize=9)
            ax.set_xlim((condition_numbers[0] / 2, MAX_CONDITION_NUMBER * 100))
            steps = int(np.log10(MAX_CONDITION_NUMBER / condition_numbers[0]))
            ax.set_xscale("log")

            ticks = np.logspace(
                np.log10(condition_numbers[0]),
                np.log10(MAX_CONDITION_NUMBER),
                steps+2,
            )
            ticklabels = [f"$10^{{{int(np.log10(tick))}}}$" for tick in ticks]
            ax.set_xticks(ticks, labels=ticklabels)
            ax.set_xlabel("Condition number $\\kappa$")
            ax.set_ylabel("Performance $m_c / m_s$")

        if row == 0:
            ax.text(
                condition_numbers[-1] * 10,
                YLIMIT[1],
                f"$\\mathbf{{\\kappa_n = 10^{{{int(np.log10(right_cluster_condition_number))}}}}}$",
                verticalalignment="bottom",
                horizontalalignment="center",
            )
        if col == 0:
            ax.text(
                condition_numbers[0]*10,
                YLIMIT[1],
                f"$\\mathbf{{\\lambda_1 = 10^{{{int(np.log10(min_eig))}}}}}$",
                verticalalignment="bottom",
                horizontalalignment="center",
            )

fig.tight_layout()

if ARGS.generate_output:
    fn = Path(__file__).name.replace("_fig.py", "")
    save_latex_figure(fn, fig)
if ARGS.show_output:
    plt.show()
