from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.optimize import fsolve
from scipy.special import lambertw
from tqdm import tqdm

from hcmsfem.cli import CLI_ARGS
from hcmsfem.plot_utils import (
    CUSTOM_COLORS_SIMPLE,
    save_latex_figure,
    set_mpl_cycler,
    set_mpl_style,
)

set_mpl_style()
set_mpl_cycler(colors=True, lines=True)

###################
# CONSTANT INPUTS #
###################
# CG convergencetas
TOLERANCE = 1e-8

# spectra
MIN_EIGS = [1e-8]  # reciprocal of the contrast of problem coefficient
# LEFT_CLUSTER_WIDTHS_MULTIPLIERS = [1e-2, 1, 1e2, 1e4, 1e6]
LEFT_CLUSTER_CONDITION_NUMBERS = [1, 1e1, 1e2, 1e3, 1e4]
RIGHT_CLUSTER_CONDITION_NUMBERS = [1e1, 1e2]  # condition number bound for contrast=1
MAX_CONDITION_NUMBER = 1e10  # maximum global condition number

# plot
FIGWIDTH = 8
FIGHEIGHT = (FIGWIDTH / len(RIGHT_CLUSTER_CONDITION_NUMBERS)) * len(MIN_EIGS)
LEGEND_HEIGHT = 0.1
RESOLUTION = int(1e3)
YLIMIT = (1e-1, 1e4)  # y-axis limits for performance plot


#############
# FUNCTIONS #
#############
# generating spectra
def get_spectra(right_cluster_condition_number: float, min_eig: float):
    # minimum global condition number
    min_condition_number = (
        LEFT_CLUSTER_CONDITION_NUMBERS[0] * right_cluster_condition_number
    )

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
    b_1s = min_eig * np.array(LEFT_CLUSTER_CONDITION_NUMBERS)
    b_n = min_eig * condition_numbers
    a_n = b_n / right_cluster_condition_number
    left_clusters = [(min_eig, b_1) for b_1 in b_1s]
    right_clusters = [(a_n[i], b_n[i]) for i in range(RESOLUTION + 1)]
    return left_clusters, right_clusters, condition_numbers


# performance calculation
def calculate_performance(
    left_clusters: list[tuple[float, float]],
    right_clusters: list[tuple[float, float]],
    condition_numbers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    m_c = classical_bound(condition_numbers)  # classical CG bound
    performance = np.zeros((len(LEFT_CLUSTER_CONDITION_NUMBERS), RESOLUTION + 1))
    theoretical_improvement = np.zeros_like(performance)
    with ThreadPoolExecutor() as executor:
        desc = "Calculating performance surface"
        futures_1 = [
            executor.submit(
                compute_bound_for_width, i, left_clusters[i], right_clusters
            )
            for i in range(len(LEFT_CLUSTER_CONDITION_NUMBERS))
        ]
        futures_2 = [
            executor.submit(
                compute_theoretical_improvement_for_width,
                i,
                left_clusters[i],
                right_clusters,
            )
            for i in range(len(LEFT_CLUSTER_CONDITION_NUMBERS))
        ]
        with tqdm(total=len(futures_1), desc=desc) as pbar:
            for future_1, future_2 in zip(futures_1, futures_2):
                i, m_s_i = future_1.result()
                performance[i, :] = m_c / m_s_i
                i, improvement_i = future_2.result()
                theoretical_improvement[i, :] = improvement_i
            pbar.update(1)

    return performance, theoretical_improvement


# classical CG bound
def classical_bound(condition_numbers: np.ndarray) -> np.ndarray:
    return np.sqrt(condition_numbers) * np.log(2 / TOLERANCE) / 2


def new_bound(clusters: list[tuple[float, float]]) -> int:
    k_l = clusters[0][1] / clusters[0][0]  # condition number of left cluster
    k_r = clusters[1][1] / clusters[1][0]  # condition number of right cluster
    s = clusters[1][1] / clusters[0][1]  # spectral gap
    log_4s = np.log(4 * s)
    sqrt_k_r = np.sqrt(k_r)
    sqrt_k_l = np.sqrt(k_l)
    return np.ceil(
        sqrt_k_r * log_4s / 2
        + np.log(2 / TOLERANCE)
        * (sqrt_k_l + sqrt_k_r + sqrt_k_l * sqrt_k_r * log_4s)
        / 2
    )


# improved CG bound
def compute_bound_for_width(i, left_cluster_i, right_clusters):
    m_s_i = np.zeros(RESOLUTION + 1)  # performance for this left cluster
    for j, rcluster in enumerate(right_clusters):
        clusters = [left_cluster_i, rcluster]
        if rcluster[0] < left_cluster_i[1]:
            # if the right cluster overlaps with the left cluster, skip
            m_s_i[j] = np.nan
            continue
        m_s = new_bound(clusters)
        m_s_i[j] = m_s
    return i, m_s_i


def compute_theoretical_improvement_for_width(i, left_cluster_i, right_clusters):
    improvement_i = np.zeros(RESOLUTION + 1)  # theoretical improvement row i
    k_l = left_cluster_i[1] / left_cluster_i[0]  # condition number of left cluster
    for j, right_cluster_j in enumerate(right_clusters):
        k = right_cluster_j[1] / left_cluster_i[0]  # global condition number
        k_r = right_cluster_j[1] / right_cluster_j[0]
        s = right_cluster_j[1] / left_cluster_i[1]  # spectral gap
        improvement_i[j] = np.sqrt(k / (k_l * k_r)) - np.log(4 * s)
    return i, improvement_i


# improved CG bound (uniform performance)
def compute_uniform_performance(
    right_clusters: list[tuple[float, float]],
    min_eig: float,
    condition_numbers: np.ndarray,
) -> np.ndarray:
    m_c = classical_bound(condition_numbers)  # classical CG bound
    uniform_performance = np.zeros((RESOLUTION + 1,))
    for i, rcluster in enumerate(right_clusters):
        lcluster = (
            min_eig,
            rcluster[0],
        )  # left cluster is almost touching the right cluster
        clusters = [lcluster, rcluster]
        m_s = new_bound(clusters)  # improved CG bound for uniform performance
        uniform_performance[i] = m_c[i] / m_s

    return uniform_performance


def compute_theoretical_improvement_boundary(
    k_r: float,
    min_eig: float,
    condition_numbers: np.ndarray,
):
    theoretical_improvement_boundary = np.zeros_like(condition_numbers)

    def func(s, k):
        log_4s = np.log(4 * s)
        sqrt_s = np.sqrt(s)
        return (
            log_4s
            + np.sqrt(1 / k_r) * (1 - sqrt_s)
            + 0 * (1 + log_4s) / (np.sqrt(k / s) * np.log(2 / TOLERANCE))
            + 1 * 1 / np.sqrt(k_r)
            + 0 * np.sqrt(s / k)
        )

    m_c = classical_bound(condition_numbers)  # classical CG bound
    for i, k in enumerate(condition_numbers):
        s = fsolve(func, 1000, args=(k,))[0]
        k_l = k / s
        left_cluster = (min_eig, min_eig * k_l)  # left cluster
        right_cluster = (min_eig * k / k_r, min_eig * k)  # right cluster
        clusters = [left_cluster, right_cluster]
        m_s = new_bound(clusters)  # improved CG bound
        theoretical_improvement_boundary[i] = m_c[i] / m_s
    return theoretical_improvement_boundary


def improvement_boundary_condition_numbers_lambert(
    k_r: float, left_clusters: list[tuple[float, float]]
) -> float:
    a = 1 / np.sqrt(k_r)  # + 1 / np.sqrt(condition_numbers)
    b = 1 / np.sqrt(k_r)
    k_ls = np.array([l[1] / l[0] for l in left_clusters])
    s: np.float64 = np.real((2 / a) ** 2 * lambertw(-a * np.exp(-b / 2) / 4, k=-1) ** 2)
    return s * k_ls


def improvement_boundary_condition_numbers_lambert_expansion(
    k_r: float, left_clusters: list[tuple[float, float]]
) -> float:
    x = -1 / (4 * np.sqrt(k_r) * np.exp(1 / (2 * np.sqrt(k_r))))
    L = np.log(-x)
    l = np.log(-L)
    k_ls = np.array([l[1] / l[0] for l in left_clusters])
    s = 4 * k_r * (L - l + l / L) ** 2
    return s * k_ls


# plot performance curves per width
def plot_performance_curves(
    ax: Axes,
    min_eig: float,
    right_cluster_condition_number: float,
    condition_numbers: np.ndarray,
    performance: np.ndarray,
    theoretical_improvement: np.ndarray,
    theoretical_improvement_boundary: np.ndarray,
    estimated_theoretical_improvement_boundary: np.ndarray,
    estimated_theoretical_improvement_boundary_approximation: np.ndarray,
    uniform_performance: np.ndarray,
):
    for i, k_l in enumerate(LEFT_CLUSTER_CONDITION_NUMBERS):
        expected_improvement = theoretical_improvement[i, :] >= 0
        # below threshold
        ax.plot(
            condition_numbers[~expected_improvement],
            performance[i, :][~expected_improvement],
            lw=2,
            linestyle="--",
            alpha=0.8,
        )

        # above threshold
        ax.plot(
            condition_numbers[expected_improvement],
            performance[i, :][expected_improvement],
            lw=2,
            linestyle="-",
            color=ax.lines[-1].get_color(),  # match color of dashed line
        )

        # condition numbers for theoretical improvement boundary (lambert W)
        k = estimated_theoretical_improvement_boundary[i]
        p = compute_theoretical_improvement_boundary(
            right_cluster_condition_number, min_eig, [k]
        )
        ax.plot(
            k,
            p,  # ~=1
            marker="x",
            color=ax.lines[-1].get_color(),  # match color of dashed line
            markersize=10,
            linestyle="None",
            label="$\\kappa$-threshold (exact)" if i == 0 else None,
            zorder=10,
        )

        # condition numbers for theoretical improvement boundary (lambert W expansion around 0)
        k = estimated_theoretical_improvement_boundary_approximation[i]
        p = compute_theoretical_improvement_boundary(
            right_cluster_condition_number, min_eig, [k]
        )
        ax.plot(
            k,
            p,  # ~=1
            marker="o",
            color=ax.lines[-1].get_color(),  # match color of dashed line
            markersize=5,
            linestyle="None",
            label="$\\kappa$-threshold (approx.)" if i == 0 else None,
            zorder=11,
        )

        # Annotate at the right end of each curve
        x_annot = condition_numbers[-1] * 10
        y_annot = performance[i, -1]
        ax.text(
            x_annot,
            y_annot,
            f"$\\mathbf{{k_l}}=10^{{{int(np.log10(k_l))}}}$",
            fontsize=8,
            va="center",
            ha="center",
            color=ax.lines[-1].get_color(),  # match curve color
            clip_on=False,
            fontweight="bold",
        )

    alpha = 0.6
    ax.fill_between(
        condition_numbers,
        uniform_performance,
        theoretical_improvement_boundary,
        color=CUSTOM_COLORS_SIMPLE[0],
        edgecolor="black",
        linestyle="--",
        linewidth=1,
        alpha=alpha,
        label="$P$-bounds",
        hatch="///",
        hatch_linewidth=1,
        zorder=9,
    )
    ax.fill_between(
        condition_numbers,
        theoretical_improvement_boundary,
        np.ones_like(condition_numbers),
        color=CUSTOM_COLORS_SIMPLE[0],
        edgecolor="None",
        alpha=alpha,
        label="No-improvement",
        linestyle="None",
        zorder=8,
    )

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
        performance, theoretical_improvement = calculate_performance(
            left_clusters, right_clusters, condition_numbers
        )
        uniform_performance = compute_uniform_performance(
            right_clusters, min_eig, condition_numbers
        )
        theoretical_improvement_boundary = compute_theoretical_improvement_boundary(
            right_cluster_condition_number, min_eig, condition_numbers
        )
        estimated_theoretical_improvement_boundary = (
            improvement_boundary_condition_numbers_lambert(
                right_cluster_condition_number, left_clusters
            )
        )
        estimated_theoretical_improvement_boundary_approximation = (
            improvement_boundary_condition_numbers_lambert_expansion(
                right_cluster_condition_number, left_clusters
            )
        )
        plot_performance_curves(
            ax,
            min_eig,
            right_cluster_condition_number,
            condition_numbers,
            performance,
            theoretical_improvement,
            theoretical_improvement_boundary,
            estimated_theoretical_improvement_boundary,
            estimated_theoretical_improvement_boundary_approximation,
            uniform_performance,
        )
        if row == 0 and col == 0:
            ax.legend(fontsize=8, loc="upper left")
            ax.set_xlim((condition_numbers[0] / 2, MAX_CONDITION_NUMBER * 100))
            steps = int(np.log10(MAX_CONDITION_NUMBER / condition_numbers[0]))
            ax.set_xscale("log")

            ticks = np.logspace(
                np.log10(condition_numbers[0]),
                np.log10(MAX_CONDITION_NUMBER),
                steps + 2,
            )
            ticklabels = [f"$10^{{{int(np.log10(tick))}}}$" for tick in ticks]
            ax.set_xticks(ticks, labels=ticklabels)
            ax.set_xlabel("Condition number $\\kappa$")
            ax.set_ylabel("Performance $P = m / \\bar{m}$")

        if row == 0:
            ax.text(
                condition_numbers[-1] * 10,
                YLIMIT[1],
                f"$\\mathbf{{\\kappa_r = 10^{{{int(np.log10(right_cluster_condition_number))}}}}}$",
                verticalalignment="bottom",
                horizontalalignment="center",
            )
        if col == 0:
            ax.text(
                condition_numbers[0] * 10,
                YLIMIT[1],
                f"$\\mathbf{{\\lambda_1 = 10^{{{int(np.log10(min_eig))}}}}}$",
                verticalalignment="bottom",
                horizontalalignment="center",
            )

fig.tight_layout()

if CLI_ARGS.generate_output:
    fn = Path(__file__).name.replace("_fig.py", "")
    save_latex_figure(fn, fig)
if CLI_ARGS.show_output:
    plt.show()
