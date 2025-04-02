import matplotlib.pyplot as plt
import numpy as np

from clibs.custom_cg import CustomCG
from utils.utils import (
    get_cli_args,
    mpl_graph_plot_style,
    save_latex_figure,
    set_mpl_cycler,
    set_mpl_style,
    CustomColors
)

# constants
ARGS = get_cli_args()
EIGV_SAMPLE_RANGE = (
    0.1,
    0.9,
)  # within which eigenvalues are sampled (condition number is max_eig/min_eig)
CLUSTER_COUNTS = [
    2,
    3,
    4,
]  # number of clusters (must divide problem size and cannot exceed n!)
MIN_PROBLEM_SIZE = np.lcm.reduce(
    CLUSTER_COUNTS
)  # minimum problem size is the LCM of cluster counts
PROBLEM_SIZE = 30 * MIN_PROBLEM_SIZE  # size of the system
CLUSTER_SPREADS = [0.02, 0.04]  # spread of eigenvalues within each cluster
NUM_RESPOLY = 3  # number of residual polynomials to plot (from the end)
RESOLUTION = 2000
DOMAIN = (0, 1)  # domain for residual polynomial plot
CODOMAIN = (-1.5, 1.5)
FIGWIDTH = 6  # inches

# set matplotlib style
set_mpl_style()
set_mpl_cycler(lines=True, markers=False, colors=True)

# set random seed
rng = np.random.default_rng(42)

# initiate problem
b = rng.random(PROBLEM_SIZE)
x0 = np.zeros(PROBLEM_SIZE)

# construct matrices with various eigenvalue distributions, but same condition number
As = []
eigs = []
clusters = []
for cluster_count in CLUSTER_COUNTS:
    cluster_size = PROBLEM_SIZE // cluster_count
    for spread in CLUSTER_SPREADS:
        eig = np.array([])
        lower_bound = EIGV_SAMPLE_RANGE[0] + spread
        upper_bound = EIGV_SAMPLE_RANGE[1] - spread
        centers = np.linspace(lower_bound, upper_bound, cluster_count, endpoint=True)
        cluster = []
        for center in centers:
            lb = center - spread
            ub = center + spread
            even_dist = np.linspace(lb, ub, cluster_size)
            eig = np.append(eig, even_dist)
            cluster.append((lb, ub))
        clusters.append(cluster)
        eig_sorted = np.sort(eig)
        As.append(np.diag(eig_sorted))
        eigs.append(eig_sorted)

# experiments
figratio = len(CLUSTER_COUNTS) / len(CLUSTER_SPREADS)  # height/width
fig, axs = plt.subplots(
    nrows=len(CLUSTER_COUNTS),
    ncols=len(CLUSTER_SPREADS),
    sharex=True,
    sharey=True,
    squeeze=False,
    figsize=(FIGWIDTH, FIGWIDTH * figratio),
)
domain_range = DOMAIN[1] - DOMAIN[0]
codomain_range = CODOMAIN[1] - CODOMAIN[0]
classical_iteration_upperbound = 0
sharpened_iteration_upperbounds = []
iterations = []
axs_texts = []
for i, A in enumerate(As):
    # select axis
    row = i // len(CLUSTER_SPREADS)
    col = i % len(CLUSTER_SPREADS)
    ax = axs[row, col]

    # solve exact solution (use diagonality of A)
    x_exact = b / eigs[i]

    # solve system
    custom_cg = CustomCG(A, b, x0)
    x, success = custom_cg.solve(
        save_iterates=False, save_search_directions=False, x_exact=x_exact
    )

    # calculate classical upperbound for the number of iterations
    if i == 0:  # upperbound is the same for all matrices
        classical_iteration_upperbound = custom_cg.calculate_iteration_upperbound()
        print(f"classical upperbound: {classical_iteration_upperbound}")

    # sharpened for the number of iterations
    sharpened_iteration_upperbounds.append(custom_cg.calculate_improved_cg_iteration_upperbound(
        clusters[i]
    ))
    print(
        f"#clusters: {CLUSTER_COUNTS[row]}, spread: {CLUSTER_SPREADS[col]}, m_p: {sharpened_iteration_upperbounds[-1]}"
    )

    # calculate residual polynomials
    cg_poly_x, cg_poly_r, cg_poly_e = custom_cg.cg_polynomial(RESOLUTION, domain=DOMAIN)
    iterations.append(len(cg_poly_r) - 1)

    # plot final residual polynomial(s)
    label = r"$r_m$"
    ax.plot(cg_poly_x, cg_poly_r[-1], label=label)
    for j in range(1, NUM_RESPOLY):
        index = -(j + 1)
        if abs(index) > len(cg_poly_r):
            continue
        label = r"$r_{m-" + f"{j}" + r"}$"
        ax.plot(cg_poly_x, cg_poly_r[index], label=label, zorder=9)

    # add eigenvalues of A
    ax.scatter(
        np.real(eigs[i]),
        np.imag(eigs[i]),
        marker=".",
        color="black",
        zorder=10,
        s=20,
    )

    # axis properties
    ax.set_ylim(CODOMAIN)
    convergence_info = f"$m$ = {iterations[-1]}"
    sigma_text = r"$\mathbf{\sigma = " + f"{CLUSTER_SPREADS[col]}" + "}$"
    n_c_text = r"$\mathbf{n_c = " + f"{CLUSTER_COUNTS[row]}" + "}$"

    axs_texts.append(ax.text(
        DOMAIN[0] + 0.5 * domain_range,
        CODOMAIN[0] + 0.9 * codomain_range,
        convergence_info,
        horizontalalignment="center",
        zorder=11,
    ))
    if row == 0:
        ax.set_title(sigma_text)
    if col == 0:
        ax.text(
            DOMAIN[0] - 0.1 * domain_range,
            0,
            n_c_text,
            verticalalignment="center",
            horizontalalignment="right",
        )

    # set classic axis style
    ax = mpl_graph_plot_style(
        ax, DOMAIN, CODOMAIN, origin=True, xtick_locs=[DOMAIN[1]], ytick_locs=CODOMAIN
    )

# plot legend last
pos_x = 0
pos_y = 0
height = 0.1 * codomain_range
width = 1.1 * domain_range
axs[0, 0].legend(
    fontsize=8,
    loc="lower center",
    ncol=NUM_RESPOLY,
    mode="expand",
    bbox_to_anchor=(pos_x, pos_y, width, height),
).set_zorder(11)

fig.tight_layout()
if ARGS.generate_output:
    save_latex_figure("effect_of_eigenvalue_distribution")
if ARGS.show_output:
    fig.show()
    input()

# add improved classical upperbound
for i, _ in enumerate(As):
    row = i // len(CLUSTER_SPREADS)
    col = i % len(CLUSTER_SPREADS)
    ax = axs[row, col]
    convergence_info = f"$m = {iterations[i]} < {sharpened_iteration_upperbounds[i]}"+ r" = \bar{m}$"
    color = CustomColors.RED if classical_iteration_upperbound < sharpened_iteration_upperbounds[i] else CustomColors.NAVY
    axs_texts[i].set_text(convergence_info)

if ARGS.generate_output:
    save_latex_figure("effect_of_eigenvalue_distribution_sharpened_bounds")
if ARGS.show_output:
    fig.show()
    input()