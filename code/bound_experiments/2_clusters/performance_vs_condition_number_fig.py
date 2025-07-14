from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.optimize import brentq, fsolve, root_scalar
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
LEFT_CLUSTER_CONDITION_NUMBERS = [1, 1e1, 1e2, 1e3, 1e4]
RIGHT_CLUSTER_CONDITION_NUMBERS = [2, 1e3]  # condition number bound for contrast=1
MAX_CONDITION_NUMBER = 1e10  # maximum global condition number

# plot
FIGWIDTH = 3.5 * len(RIGHT_CLUSTER_CONDITION_NUMBERS)
FIGHEIGHT = (FIGWIDTH / len(RIGHT_CLUSTER_CONDITION_NUMBERS)) * len(MIN_EIGS)
LEGEND_HEIGHT = 0.1
RESOLUTION = int(1e3)
YLIMIT = (5e-2, 1e4)  # y-axis limits for performance plot


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


# classical CG iteration bound (approximate variant)
def classical_bound(condition_numbers: np.ndarray) -> np.ndarray:
    return np.sqrt(condition_numbers) * np.log(2 / TOLERANCE) / 2

# two-cluster CG iteration bound 
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
        if s <= 0:
            return np.nan
        log_4s = np.log(4 * s)
        sqrt_s = np.sqrt(s)
        return (
            log_4s
            + np.sqrt(1 / k_r) * (1 - sqrt_s)
            + 0 * (1 + log_4s) / (np.sqrt(k / s) * np.log(2 / TOLERANCE)) # neglected term
            + 1 * 1 / np.sqrt(k_r)
            + 0 * np.sqrt(s / k) # neglected term
        )

    def solve_robust(func, k, method="brentq"):
        """
        Robust solver that tries multiple methods and initial guesses
        """
        if method == "brentq":
            # Brent's method - guaranteed convergence if root exists in bracket
            # Need to find a bracketing interval first
            try:
                # Try to find reasonable bounds - ensure s > 0
                s_min = 1
                s_max = 1e3

                # Extend bounds if needed to ensure opposite signs
                max_iterations = 50  # Increase iterations for larger ranges
                iterations = 0
                while (
                    func(s_min, k) * func(s_max, k) > 0 and iterations < max_iterations
                ):
                    s_max = min(s_max * 10, 1e16)  # Don't go above 1e16
                    iterations += 1

                if func(s_min, k) * func(s_max, k) <= 0:
                    return brentq(func, s_min, s_max, args=(k,))
                else:
                    raise ValueError("Could not find bracketing interval")
            except Exception as e:
                # Fall back to other methods if bracketing fails
                raise ValueError(f"Bracketing failed for k={k}: {e}")

        if method == "root_scalar" or method == "hybrid":
            # Try root_scalar with different methods
            methods_to_try = ["brentq", "brenth", "ridder", "bisect"]

            for solver_method in methods_to_try:
                try:
                    # Find bracketing interval - ensure s > 0
                    s_min = 1e-6
                    s_max = 1e6

                    # Extend bounds if needed
                    max_iterations = 20
                    iterations = 0
                    while (
                        func(s_min, k) * func(s_max, k) > 0
                        and iterations < max_iterations
                    ):
                        s_min = max(s_min / 10, 1e-12)  # Don't go below 1e-12
                        s_max = min(s_max * 10, 1e12)  # Don't go above 1e12
                        iterations += 1

                    if func(s_min, k) * func(s_max, k) <= 0:
                        result = root_scalar(
                            func,
                            args=(k,),
                            bracket=[s_min, s_max],
                            method=solver_method,
                        )
                        if result.converged:
                            return result.root
                except:
                    continue

        # Fall back to fsolve with multiple initial guesses - all positive
        print("Falling back to fsolve with multiple initial guesses")
        initial_guesses = [1, 10, 100, 250, 1000, 0.1, 0.01, 2500]
        for guess in initial_guesses:
            try:
                result = fsolve(func, guess, args=(k,), full_output=True)
                if (
                    result[2] == 1 and result[0][0] > 0
                ):  # Check if converged and positive
                    return result[0][0]
            except:
                continue

        # If all else fails, use the original approach
        try:
            result = fsolve(func, 250, args=(k,))[0]
            if result > 0:
                return result
            else:
                # If negative, try a different guess
                return fsolve(func, 1000, args=(k,))[0]
        except:
            # Last resort - return a reasonable default
            return 1.0

    m_c = classical_bound(condition_numbers)  # classical CG bound
    for i, k in enumerate(condition_numbers):
        s = solve_robust(func, k, method="brentq")
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
    a = 1 / np.sqrt(k_r)
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
    # No-improvement region
    alpha = 0.6
    ax.fill_between(
        condition_numbers,
        uniform_performance,
        np.ones_like(condition_numbers),
        color=CUSTOM_COLORS_SIMPLE[0],
        edgecolor="None",
        alpha=alpha,
        label="$P_{\\text{uniform}} \\leq P \\leq 1$",
        linestyle="None",
        zorder=8,
    )

    # P-bounds
    ax.fill_between(
        condition_numbers,
        uniform_performance,
        theoretical_improvement_boundary,
        color="None",
        edgecolor="black",
        linestyle="--",
        linewidth=1,
        alpha=alpha,
        label="$P_{\\text{uniform}} \\leq P \\leq P_{\\text{threshold}}$",
        hatch="///",
        hatch_linewidth=1,
        zorder=9,
    )

    # approximate minimum performance
    r_log_4k_r = 1 / np.log(4 * right_cluster_condition_number)
    min_p = r_log_4k_r - r_log_4k_r**2 * (
        np.sqrt(right_cluster_condition_number / condition_numbers)
        + 1 / np.sqrt(right_cluster_condition_number)
        + np.sqrt(right_cluster_condition_number / condition_numbers)
        / np.log(2 / TOLERANCE)
        + 2 / (np.sqrt(condition_numbers) * np.log(2 / TOLERANCE))
    )
    ax.plot(
        condition_numbers,
        min_p * np.ones_like(condition_numbers),
        color="red",
        linestyle=":",
        linewidth=2,
        alpha=0.8,
        dashes=(3, 4),  # (dash_length, gap_length) in points
        label="$P^{(1)}_{\\text{uniform}}$",
        zorder=10,
    )

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
            label="$T_{\\kappa}$" if i == 0 else None,
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
            label="$T^{(0)}_{\\kappa}$" if i == 0 else None,
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
        if row == 0 and col == 1:
            ax.legend(fontsize=9, loc="upper left", framealpha=0.7, shadow=False)
        if row == 0 and col == 0:
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
            ax.set_ylabel("Performance $P = m / m_2$")

        if row == 0:
            exponent = int(np.log10(right_cluster_condition_number))
            if exponent == 0:
                kappa_r_text = (
                    f"$\\mathbf{{\\kappa_r = {int(right_cluster_condition_number)}}}$"
                )
            else:
                kappa_r_text = f"$\\mathbf{{\\kappa_r = 10^{{{exponent}}}}}$"
            ax.text(
                condition_numbers[-1] * 10,
                YLIMIT[1],
                kappa_r_text,
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

        # turn on grid
        ax.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)

fig.tight_layout()

if CLI_ARGS.generate_output:
    fn = Path(__file__).name.replace("_fig.py", "")
    save_latex_figure(fn, fig)
if CLI_ARGS.show_output:
    plt.show()
