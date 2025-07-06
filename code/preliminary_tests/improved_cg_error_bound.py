from math import log as math_log
from math import sqrt as math_sqrt

import matplotlib.pyplot as plt
import numpy as np

from hcmsfem.plot_utils import (
    MPL_COLORS,
    mpl_graph_plot_style,
    set_mpl_cycler,
    set_mpl_style,
)

# constants
FIGWIDTH = 6
LEGEND_HEIGHT = 0.1
DOMAIN = (0, 10)
CODOMAIN = (-1, 1)
ZOOMFIT = True
RESOLUTION = 5000

# problem variables
spread = 0.3
# cluster1 = (0.1, 0.1 + 2 * spread)
# cluster2 = (0.9 - 2 * spread, 0.9)
cluster1 = (1e-7, 2e-7)
cluster2 = (2 * 1e-3, 6)
rtol = 1e-6


# pure chebyshev polynomial
def chebyshev_polynomial(
    z: np.ndarray | float | int | complex, n: int
) -> np.ndarray | float:
    if n == 0:
        raise ValueError("n must be greater than 0")
    out = (z + np.emath.sqrt(z**2 - 1)) ** n + (z - np.emath.sqrt(z**2 - 1)) ** n  # type: ignore
    return out / 2


# scaled chebyshev
def scaled_chebyshev_polynomial(
    z: np.ndarray | float | int | complex, n: int, interval: tuple
) -> np.ndarray | float:
    if len(interval) != 2:
        raise ValueError("interval must be a tuple of length 2")
    if interval[0] >= interval[1]:
        raise ValueError("interval must be a tuple of increasing values")
    x1, x2 = interval
    # though correct, code below causes overflow due to high values of chebpolys
    num = chebyshev_polynomial((x2 + x1 - 2 * z) / (x2 - x1), n)
    denom = chebyshev_polynomial((x2 + x1) / (x2 - x1), n)

    return num / denom


# log of approximation for z outside [-1, 1]
def log_approx_chebyshev_polynomial(z: np.ndarray, n: int) -> np.ndarray | float:
    approx = np.zeros_like(z)
    z_test1, z_test2 = 0.0, 0.0
    if isinstance(z, (int, float, complex)):
        z_test1, z_test2 = z, z
    else:
        z_test1, z_test2 = z[0], z[-1]
    if z_test1 > 1:
        approx = n * np.log(z + np.sqrt(z**2 - 1))
    elif z_test2 < -1:
        approx = n * np.log(np.abs(z - np.sqrt(z**2 - 1)))
    else:
        raise ValueError("z must be outside the interval [-1, 1]")

    return approx - np.log(2)


def log_scaled_approx_chebyshev_polynomial(
    z: np.ndarray, n: int, interval: tuple
) -> np.ndarray | float:
    if len(interval) != 2:
        raise ValueError("interval must be a tuple of length 2")
    if interval[0] >= interval[1]:
        raise ValueError("interval must be a tuple of increasing values")
    x1, x2 = interval
    return log_approx_chebyshev_polynomial(
        (x2 + x1 - 2 * z) / (x2 - x1), n
    ) - log_approx_chebyshev_polynomial((x2 + x1) / (x2 - x1), n)


def complex_abs(z: np.ndarray | float | int | complex) -> np.ndarray:
    return np.sqrt(np.real(z) ** 2 + np.imag(z) ** 2)


def residual_polynomial_bound_for_few_eigenvalues(
    z: np.ndarray, m: int, interval: tuple
) -> float:
    if len(interval) != 2:
        raise ValueError("interval must be a tuple of length 2")
    if interval[0] >= interval[1]:
        raise ValueError("interval must be a tuple of increasing values")
    x1, x2 = interval
    return (x2 / x1 - 1) ** m


def calculate_chebyshev_degree(cond: float, log_rtol: float = math_log(1e-6)) -> int:
    convergence_factor = (np.sqrt(cond) - 1) / (np.sqrt(cond) + 1)
    return int(np.ceil((log_rtol - np.log(2)) / np.log(convergence_factor)))


def calculate_chebyshev_degrees(
    clusters: list[tuple[float, float]],
    rtol: float = 1e-6,
) -> list[int]:
    # setup
    log_rtol = math_log(rtol)
    degrees = [0] * len(clusters)

    for i, cluster in enumerate(clusters):
        a_i, b_i = cluster
        log_rtol_eff = log_rtol
        for j in range(i):
            a_j, b_j = clusters[j]
            z_1 = (b_j + a_j - 2 * b_i) / (b_j - a_j)
            z_2 = (b_j + a_j) / (b_j - a_j)
            m_j = degrees[j]
            log_rtol_eff -= m_j * (
                math_log(
                    abs(z_1 - math_sqrt(z_1**2 - 1)) / (z_2 + math_sqrt(z_2**2 - 1))
                )
            )

        # calculate & store chebyshev degree
        degrees[i] = calculate_chebyshev_degree(b_i / a_i, log_rtol=log_rtol_eff)

    return degrees


def calculate_all_respoly_errors(
    intervals: list[tuple[float, float]],
    degrees: list[int],
    m_th: int,
    resolution: int = 100,
):
    spectrum = (intervals[0][0], intervals[-1][1])
    z = np.linspace(*spectrum, resolution)

    max_err_single_chebyshev = np.max(
        complex_abs(scaled_chebyshev_polynomial(z, m_th, spectrum))
    )
    max_err_seperate_chebyshevs = [0.0] * len(intervals)
    max_err_product_chebyshevs = -np.inf
    for i, interval_i in enumerate(intervals):
        z = np.linspace(*interval_i, resolution)
        m_i = degrees[i]

        # chebyshev on this interval_i
        C = np.log(complex_abs(scaled_chebyshev_polynomial(z, m_i, interval_i)))

        # max error for single chebyshev on this interval_i
        max_err_seperate_chebyshevs[i] = np.max(C)

        # max error for product of chebyshevs on this interval_i
        degrees_copy = degrees.copy()
        intervals_copy = intervals.copy()
        degrees_copy.pop(i)
        intervals_copy.pop(i)
        for interval_j, m_j in zip(intervals_copy, degrees_copy):
            C += log_scaled_approx_chebyshev_polynomial(
                z, m_j, interval_j
            )  # complex_abs(scaled_chebyshev_polynomial(z, m_j, interval_j))
        err_product = np.max(C)
        print(err_product)
        if err_product > max_err_product_chebyshevs:
            max_err_product_chebyshevs = err_product

    print(f"max_{spectrum}|C_{m_th}|: {max_err_single_chebyshev:2e}")
    for i, (interval, degree) in enumerate(zip(intervals, degrees)):
        print(f"log(max_{interval}|C_{degree}|): {max_err_seperate_chebyshevs[i]:2e}")
    C_product_str = "C_" + "*C_".join(str(degree) for degree in degrees)
    max_str = "max_[" + ",".join(str(interval) for interval in intervals) + "]"
    print(f"{max_str}|{C_product_str}|: {np.exp(max_err_product_chebyshevs):2e}")


# chebyshev degree calculation (whole spectrum)
m_th = calculate_chebyshev_degree(cluster2[1] / cluster1[0], log_rtol=math_log(rtol))
[m1, m2] = calculate_chebyshev_degrees([cluster1, cluster2], rtol=rtol)

# chebyshev polynomials as residual polynomials
z = np.linspace(*DOMAIN, RESOLUTION)
C = complex_abs(scaled_chebyshev_polynomial(z, m_th, (cluster1[0], cluster2[1])))
C_1_abs = complex_abs(scaled_chebyshev_polynomial(z, m1, (cluster1[0], cluster1[1])))
C_2_abs = complex_abs(scaled_chebyshev_polynomial(z, m2, (cluster2[0], cluster2[1])))
C_1_2_abs = C_1_abs * C_2_abs

# calculate residual polynomial error TODO: make this resolution independent
calculate_all_respoly_errors([cluster1, cluster2], [m1, m2], m_th, resolution=100)

print(f"Iteration bound sharpened by {(1 - (m1 + m2)/m_th) * 100:.2f}%.")

# figure setup
domain_size = DOMAIN[1] - DOMAIN[0]
codomain_size = CODOMAIN[1] - CODOMAIN[0]
figratio = (1 + LEGEND_HEIGHT) * codomain_size / domain_size
figheight = FIGWIDTH if ZOOMFIT else FIGWIDTH * figratio
fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH, figheight))

# axes
x_ticks = np.linspace(*DOMAIN, 6)[1:]
y_ticks = np.linspace(*CODOMAIN, 6)
set_mpl_style()
set_mpl_cycler(lines=True, markers=False, colors=True)
mpl_graph_plot_style(
    ax,
    domain=DOMAIN,
    codomain=CODOMAIN,
    origin=True,
    xtick_locs=x_ticks,
    ytick_locs=y_ticks,
)
ax.set_ylim(*CODOMAIN)
ax.set_xlabel("z")
ax.set_ylabel("C(z)")
ax.set_title(f"Chebpolys on clusters, $m_1 = {m1}$, $m_2 = {m2}$")

# plot clusters
ax.plot(
    cluster1,
    [0, 0],
    color="black",
    linestyle="-",
    linewidth=2,
    marker="|",
    markersize=10,
)
ax.plot(
    cluster2,
    [0, 0],
    color="black",
    linestyle="-",
    linewidth=2,
    marker="|",
    markersize=10,
)
ax.text(
    (cluster1[0] + cluster1[1]) / 2,
    -0.1,
    r"$\mathbf{[a,b]}$",
    fontsize=12,
    ha="center",
    va="top",
)
ax.text(
    (cluster2[0] + cluster2[1]) / 2,
    -0.1,
    r"$\mathbf{[c,d]}$",
    fontsize=12,
    ha="center",
    va="top",
)

z = np.linspace(*DOMAIN, RESOLUTION)

# plot (scaled) Chebyshev polynomial over the entire spectrum
ax.plot(
    z,
    C,
    label=r"$|C_{[a,d]," + f"{m_th}" + r"}(z)|$",
    linestyle="-",
    color=MPL_COLORS[4],
)

# plot (scaled) chebyshev polynomials
ax.plot(
    z,
    C_1_abs,
    label=r"$|C_{[a,b]," + f"{m1}" + r"}(z)|$",
    linestyle="-",
    color=MPL_COLORS[0],
)

ax.plot(
    z,
    C_2_abs,
    label=r"$|C_{[c,d]," + f"{m2}" + r"}(z)|$",
    linestyle="-",
    color=MPL_COLORS[1],
)

# plot product of Chebyshev polynomials
ax.plot(
    z,
    C_1_2_abs,
    label=r"$|C_{[a,b]," + f"{m1}" + r"}(z)| \cdot |C_{[c,d]," + f"{m2}" + r"}(z)|$",
    linestyle="--",
    color=MPL_COLORS[2],
)

# legend
legend_coords = (0.5, 0.25)
ax.legend(
    fontsize=8,
    loc="center",  # 	"lower left",
    ncol=2,
    bbox_to_anchor=legend_coords,
)

# show plot
plt.tight_layout()
plt.show()
