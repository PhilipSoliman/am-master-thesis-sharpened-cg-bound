import matplotlib.pyplot as plt
import numpy as np

# from clibs.custom_cg import CustomCG
from lib.solvers import CustomCG
from lib.utils import (
    get_cli_args,
)

# CONSTANT INPUTS
ARGS = get_cli_args()
FIGWIDTH = 5
LEGEND_HEIGHT = 0.1
DOMAIN = (0, 20)
CODOMAIN = (-1, 1)
ZOOMFIT = True

# CG convergence parameters
TOLERANCE = 1e-6

# cluster parameters
MIN_EIG = 1e-8
INITIAL_CONDITION_NUMBER = 1e8
RIGHT_CLUSTER_CONDITION_NUMBER = 10
LEFT_CLUSTER_INITIAL_WIDTH = 1e-8
LEFT_CLUSTER_MAX_WIDTH = 1e-2
LEFT_CLUSTER_WIDTH_MULTIPLIER = 2

# initial clusters
a_1 = MIN_EIG
b_1 = a_1 + LEFT_CLUSTER_INITIAL_WIDTH
b_n = MIN_EIG * INITIAL_CONDITION_NUMBER
a_n = b_n / RIGHT_CLUSTER_CONDITION_NUMBER


# spectral width (constant)
def calculate_normalized_spectral_width(
    lcluster: tuple[float, float], rcluster: tuple[float, float]
) -> float:
    a_1, b_1 = lcluster
    _, b_n = rcluster
    c_1 = (a_1 + b_1) / 2
    w_1 = b_1 - a_1
    return (b_n - c_1) / w_1


normalized_spectral_width = calculate_normalized_spectral_width((a_1, b_1), (a_n, b_n))
print(f"Normalized spectral width: {normalized_spectral_width:.2e}")

# construct clusters
left_clusters = [(a_1, b_1)]
right_clusters = [(a_n, b_n)]
max_i = int(
    np.ceil(
        np.log(LEFT_CLUSTER_MAX_WIDTH / LEFT_CLUSTER_INITIAL_WIDTH)
        / np.log(LEFT_CLUSTER_WIDTH_MULTIPLIER)
    )
)
for i in range(1, max_i + 1):
    w_i = LEFT_CLUSTER_INITIAL_WIDTH * LEFT_CLUSTER_WIDTH_MULTIPLIER**i
    b_1_i = a_1 + w_i
    c_1_i = (a_1 + b_1_i) / 2
    b_n_i = c_1_i + w_i * normalized_spectral_width
    a_n_i = b_n_i / RIGHT_CLUSTER_CONDITION_NUMBER
    left_clusters.append((a_1, b_1_i))
    right_clusters.append((a_n_i, b_n_i))
print(f"Left clusters: {left_clusters}")
print(f"Right cluster: {right_clusters}")

# CG bounds
log_rtol = np.log(TOLERANCE)
for i, (lcluster, rcluster) in enumerate(zip(left_clusters, right_clusters)):
    spectral_gap = calculate_normalized_spectral_width(lcluster, rcluster)
    condition_number = rcluster[1] / lcluster[0]
    m_c = CustomCG.calculate_iteration_upperbound_static(
        condition_number, log_rtol, exact_convergence=True
    )
    m_i = CustomCG.calculate_improved_cg_iteration_upperbound_static(
        [lcluster, rcluster], tol=TOLERANCE, exact_convergence=True
    )
    performance = m_c / m_i
    print(f"Iteration bound for cluster {i+1} (cond: {condition_number:.2e}): m_c {m_c}, m_i {m_i}, spg {spectral_gap:.2e} Performance: {performance:.2e}")