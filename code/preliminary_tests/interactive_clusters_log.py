import functools as ft
from math import log as math_log
from math import sqrt as math_sqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.widgets import Slider

from hcmsfem.plot_utils import set_mpl_cycler, set_mpl_style

# turn on interactive mode if necessary
if not plt.isinteractive():
    plt.ion()

# constants
FIGWIDTH = 6
LEGEND_HEIGHT = 0.1
DOMAIN = (1e-7, 4)
CODOMAIN = (-1, 1)
ZOOMFIT = True
RESOLUTION = int(1e4)
NUM_CLUSTERS = 2
SPREAD = 0.05
MIN_X = 1e-7
MINCLUSTER_WIDTH = 1e-7


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
            C += log_scaled_approx_chebyshev_polynomial(z, m_j, interval_j)
        err_product = np.max(C)
        if err_product > max_err_product_chebyshevs:
            max_err_product_chebyshevs = err_product

    print(f"max_[spectrum]|C_{m_th}|: {max_err_single_chebyshev:2e}")
    for i, degree in enumerate(degrees):
        print(
            f"log(max_[a_{i},b_{i}]|C_{degree}|): {max_err_seperate_chebyshevs[i]:2e}"
        )
    C_product_str = "C_" + "*C_".join(str(degree) for degree in degrees)
    max_str = "max_[" + ",".join(f"(a_{i},b_{i})" for i in range(len(intervals))) + "]"
    print(f"{max_str}|{C_product_str}|: {np.exp(max_err_product_chebyshevs):2e}")


# supporting functions
def get_clusters(num_clusters: int, spread: float) -> list[tuple[float, float]]:
    min_x = DOMAIN[0] if DOMAIN[0] != 0 else MIN_X
    centers = np.linspace(min_x + spread, DOMAIN[1] - spread, num_clusters)
    clusters = []
    for center in centers:
        clusters.append((center - spread, center + spread))
    return clusters


def plot_clusters(
    ax, num_clusters: int = NUM_CLUSTERS, spread: float = SPREAD
) -> tuple:
    clusters = get_clusters(num_clusters, spread)
    lines = []
    labels = []
    for i, cluster in enumerate(clusters):
        center = (cluster[0] + cluster[1]) / 2
        cluster_line = ax.plot(
            cluster,
            [0, 0],
            color="black",
            linestyle="-",
            linewidth=2,
            marker="|",
            markersize=10,
        )[0]
        cluster_text = ax.text(
            center,
            -0.1,
            r"$\mathbf{[" + f"a_{i},b_{i}" + "]}$",
            fontsize=12,
            ha="center",
            va="top",
        )
        lines.append(cluster_line)
        labels.append(cluster_text)
    return clusters, lines, labels


class PlotUpdate:

    def __init__(
        self,
        fig: Figure,
        clusters: list[tuple[float, float]],
        cluster_lines: list[Line2D],
        cluster_texts: list[Text],
        C_spectrum_line: Line2D,
        C_product_line: Line2D,
    ):
        self.fig = fig
        self.clusters = clusters
        self.cluster_lines = cluster_lines
        self.cluster_labels = cluster_texts
        self.C_spectrum_line = C_spectrum_line
        self.C_product_line = C_product_line

    def update_left(
        self,
        idx: int,
        val: float,
    ):
        if not self.check_left(idx, val):
            return
        self.clusters[idx] = (val, val + self.clusters[idx][1] - self.clusters[idx][0])
        self.redraw(idx)

    def update_width(
        self,
        idx: int,
        val: float,
    ):
        if not self.check_width(idx, val):
            return
        self.clusters[idx] = (self.clusters[idx][0], self.clusters[idx][0] + val)
        self.redraw(idx)

    def redraw(
        self,
        idx: int,
    ):
        self.cluster_lines[idx].set_xdata(self.clusters[idx])
        center = (self.clusters[idx][0] + self.clusters[idx][1]) / 2
        self.cluster_labels[idx].set_position((center, -0.1))
        self.update_chebyshev_polys()
        fig.canvas.draw_idle()

    def check_left(self, idx: int, val: float) -> bool:
        width = self.clusters[idx][1] - self.clusters[idx][0]
        # check for right boundary
        if idx == len(self.clusters) - 1:
            if val >= DOMAIN[1] - width:
                return False

        # check for left neighbour collision
        if idx > 0:
            midx = idx - 1
            if val <= self.clusters[midx][1]:
                return False

        # check for right neighbour collision
        if idx < len(self.clusters) - 1:
            pidx = idx + 1
            if val >= self.clusters[pidx][0] - width:
                return False
        return True

    def check_width(self, idx: int, val: float) -> bool:
        # check for right neighbour collision
        if idx < len(self.clusters) - 1:
            midx = idx + 1
            if val >= self.clusters[midx][0] - self.clusters[idx][0]:
                return False
        # check for right boundary
        elif val >= DOMAIN[1] - self.clusters[idx][0]:
            return False
        return True

    def update_chebyshev_polys(self):
        print("\nUpdating Chebyshev degrees...")
        a_min, b_max = self.clusters[0][0], self.clusters[-1][1]
        cond = b_max / a_min
        m_th = calculate_chebyshev_degree(cond)
        degrees = calculate_chebyshev_degrees(self.clusters)

        # calculate new chebyshev polynomials
        C_spectrum = complex_abs(scaled_chebyshev_polynomial(z, m_th, (a_min, b_max)))
        C_log_product = 0
        for degree, cluster in zip(degrees, self.clusters):
            C_log_product += log_scaled_approx_chebyshev_polynomial(
                z, degree, (cluster[0], cluster[1])
            )
        C_product = np.exp(C_log_product)

        # redraw chebyshev polynomials in fig
        self.C_spectrum_line.set_ydata(C_spectrum)
        C_spectrum_label = "$C_{" + f"{m_th}" + "}$"
        self.C_spectrum_line.set_label(C_spectrum_label)

        self.C_product_line.set_ydata(C_product)
        C_product_label = (
            "$C_" + "C_".join("{" + f"{degree}" + "}" for degree in degrees) + "$"
        )
        self.C_product_line.set_label(C_product_label)

        # remove old legend
        ax = self.fig.get_axes()[0]
        ax.get_legend().remove()

        # redraw legend
        legend_coords = (0.5, 0.25)
        ax.legend(
            fontsize=12,
            loc="center",  # 	"lower left",
            ncol=2,
            bbox_to_anchor=legend_coords,
        )

        # update title
        improvement = (1 - sum(degrees) / m_th) * 100
        ax.set_title(
            f"Chebpolys on clusters (CBND: {m_th}, SBND: {sum(degrees)} -> improvement: {improvement:.2f}%)"
        )

        # redraw fig
        self.fig.canvas.draw_idle()

        # calculate residual polynomial error
        calculate_all_respoly_errors(self.clusters, degrees, m_th)


# main figure setup
domain_size = DOMAIN[1] - DOMAIN[0]
codomain_size = CODOMAIN[1] - CODOMAIN[0]
figratio = (1 + LEGEND_HEIGHT) * codomain_size / domain_size
figheight = FIGWIDTH if ZOOMFIT else FIGWIDTH * figratio
fig = plt.figure(figsize=(FIGWIDTH, figheight))
ax = fig.add_subplot(111)
# x_ticks = np.linspace(*DOMAIN, 6)[1:]
# y_ticks = np.linspace(*CODOMAIN, 6)
set_mpl_style()
set_mpl_cycler(lines=True, markers=False, colors=True)
# mpl_graph_plot_style(
#     ax,
#     domain=DOMAIN,
#     codomain=CODOMAIN,
#     origin=True,
#     xtick_locs=x_ticks,
#     ytick_locs=y_ticks,
# )
ax.set_ylim(*CODOMAIN)
ax.set_xlabel("z")
ax.set_ylabel("C(z)")

# plot clusters
clusters, cluster_lines, cluster_labels = plot_clusters(ax)

# plot initial chebyshev polynomials
cond = clusters[-1][1] / clusters[0][0]
m_th = calculate_chebyshev_degree(cond)
degrees = calculate_chebyshev_degrees(clusters)
z = np.linspace(*DOMAIN, RESOLUTION)
C_spectrum = complex_abs(
    scaled_chebyshev_polynomial(z, m_th, (clusters[0][0], clusters[-1][1]))
)
C_log_product = 0
for degree, cluster in zip(degrees, clusters):
    C_log_product += log_scaled_approx_chebyshev_polynomial(
        z, degree, (cluster[0], cluster[1])
    )
C_product = np.exp(C_log_product)
C_spectrum_line = ax.semilogx(
    z, C_spectrum, label="$C_{" + f"{m_th}" + "}$", alpha=0.7
)[0]
C_product_str = "$C_" + "C_".join("{" + f"{degree}" + "}" for degree in degrees) + "$"
C_product_line = ax.semilogx(z, C_product, label=f"{C_product_str}", alpha=0.7)[0]
improvement = (1 - sum(degrees) / m_th) * 100
ax.set_title(
    f"Chebpolys on clusters (CBND: {m_th}, SBND: {sum(degrees)} -> improvement: {improvement:.2f}%)"
)

# legend
legend_coords = (0.5, 0.25)
ax.legend(
    fontsize=12,
    loc="center",  # 	"lower left",
    ncol=2,
    bbox_to_anchor=legend_coords,
)


# control figure setup
cfig_width = 6
cfig_height = 0.5 * NUM_CLUSTERS
cfig, axs = plt.subplots(NUM_CLUSTERS, 2, figsize=(cfig_width, 0.5 * NUM_CLUSTERS))
cfig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1, wspace=1, hspace=0.4)
callback = PlotUpdate(
    fig, clusters, cluster_lines, cluster_labels, C_spectrum_line, C_product_line
)
sliders = []
for i in range(NUM_CLUSTERS):
    slidermin = None
    if i > 0:
        slidermin = sliders[-2]
    ax = axs[i, :]
    sliders.append(
        Slider(
            ax=ax[0],
            label=f"$a_{i}$",
            valmin=DOMAIN[0] if DOMAIN[0] != 0 else MIN_X,
            # valmax=DOMAIN[1] - MINCLUSTER_WIDTH,
            valmax=0.1,
            valinit=clusters[i][0],
            valstep=MIN_X,
            slidermin=slidermin,
            orientation="horizontal",
        )
    )
    sliders[-1].on_changed(ft.partial(callback.update_left, i))

    sliders.append(
        Slider(
            ax=ax[1],
            label=f"$b_{i} - a_{i}$",
            valmin=MINCLUSTER_WIDTH,
            valstep=MIN_X,
            valmax=DOMAIN[1],
            valinit=2 * SPREAD,
            orientation="horizontal",
        )
    )
    sliders[-1].on_changed(ft.partial(callback.update_width, i))

fig.tight_layout()
fig.show()
input("Press Enter to exit...")
