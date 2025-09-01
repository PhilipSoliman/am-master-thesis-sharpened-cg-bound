import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from hcmsfem.cli import get_cli_args
from hcmsfem.plot_utils import (
    CustomColors,
    mpl_graph_plot_style,
    save_latex_figure,
    set_mpl_cycler,
    set_mpl_style,
)
from hcmsfem.solvers import CustomCG, generalized_cg_iteration_bound

CLI_ARGS = get_cli_args()

# constants
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
LEGEND_SIZE = 0.05  # inches
FIGRATIO = len(CLUSTER_COUNTS) / len(CLUSTER_SPREADS)  # height/width
FIGHEIGHT = FIGWIDTH * FIGRATIO + LEGEND_SIZE  # inches
FONTSIZE = 10

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

# setup figure
fig = plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
gs = gridspec.GridSpec(
    len(CLUSTER_COUNTS) + 1,
    len(CLUSTER_SPREADS),
    height_ratios=[FIGWIDTH] * len(CLUSTER_COUNTS) + [LEGEND_SIZE],
    hspace=0.15,
)
axs = []
ax_0 = fig.add_subplot(gs[0, 0])
for i in range(len(CLUSTER_COUNTS)):
    mesh_axs = []
    for j in range(len(CLUSTER_SPREADS)):
        if i == 0 and j == 0:
            mesh_axs.append(ax_0)
        else:
            mesh_axs.append(fig.add_subplot(gs[i, j], sharex=ax_0, sharey=ax_0))
    axs.append(mesh_axs)
axs = np.array(axs)
legend_ax = fig.add_subplot(gs[-1, :])
legend_ax.axis("off")
fig.subplots_adjust(hspace=0.1, left=0.12, right=0.97, top=0.95, bottom=0.02)

# experiments
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
    x, success = custom_cg.solve(x_exact=x_exact)

    # calculate classical upperbound for the number of iterations
    if i == 0:  # upperbound is the same for all matrices
        classical_iteration_upperbound = custom_cg.calculate_iteration_upperbound()
        print(f"classical upperbound: {classical_iteration_upperbound}")

    # sharpened for the number of iterations
    sharpened_iteration_upperbounds.append(
        generalized_cg_iteration_bound(
            clusters[i], log_rtol=np.log(custom_cg.tol), exact_convergence=True
        )
    )
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

    axs_texts.append(
        ax.text(
            DOMAIN[0] + 0.5 * domain_range,
            CODOMAIN[0] + 0.9 * codomain_range,
            convergence_info,
            horizontalalignment="center",
            zorder=11,
        )
    )
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
handles, labels = axs[0, 0].get_legend_handles_labels()
legend_ax.legend(
    handles,
    labels,
    fontsize=FONTSIZE,
    loc="center",
    ncol=NUM_RESPOLY,
    mode="expand",
    frameon=False,
)

if CLI_ARGS.generate_output:
    save_latex_figure("effect_of_eigenvalue_distribution")
if CLI_ARGS.show_output:
    fig.show()
    input("Press Enter to continue...")

# add improved classical upperbound
for i, _ in enumerate(As):
    row = i // len(CLUSTER_SPREADS)
    col = i % len(CLUSTER_SPREADS)
    ax = axs[row, col]
    convergence_info = (
        f"$m = {iterations[i]} < {sharpened_iteration_upperbounds[i]}" + r" = m_2$"
    )
    color = (
        CustomColors.RED.value
        if classical_iteration_upperbound < sharpened_iteration_upperbounds[i]
        else CustomColors.NAVY.value
    )
    axs_texts[i].set(text=convergence_info, color=color)

if CLI_ARGS.generate_output:
    save_latex_figure("effect_of_eigenvalue_distribution_sharpened_bounds")
if CLI_ARGS.show_output:
    fig.show()
    input("Press Enter to exit...")
