import matplotlib.pyplot as plt
import numpy as np

from lib.solvers import CustomCG
from lib.utils import (
    get_cli_args,
    mpl_graph_plot_style,
    save_latex_figure,
    set_mpl_cycler,
    set_mpl_style,
)

# constants
ARGS = get_cli_args()
EIGV_SAMPLE_RANGE = (
    0.1,
    0.9,
)  # within which eigenvalues are sampled (condition number is max_eig/min_eig)
CLUSTER_COUNTS = [
    2,
    360,
]  # number of clusters (must divide problem size and cannot exceed n!)
MIN_PROBLEM_SIZE = np.lcm.reduce(
    CLUSTER_COUNTS
)  # minimum problem size is the LCM of cluster counts
PROBLEM_SIZE = MIN_PROBLEM_SIZE  # size of the system
CLUSTER_SPREAD = 0  # spread of eigenvalues within each cluster
NUM_RESPOLY = 3  # number of residual polynomials to plot (from the end)
RESOLUTION = 2000
DOMAIN = (0, 1)  # domain for residual polynomial plot
CODOMAIN = (-1.5, 1.5)
FIGWIDTH = 5  # inches

# set matplotlib style
set_mpl_style()
set_mpl_cycler(lines=True, markers=False, colors=True)

# set random seed
rng = np.random.default_rng(42)

# initiate problem
b = rng.random(PROBLEM_SIZE)
x0 = np.zeros(PROBLEM_SIZE)
r0_norm = np.linalg.norm(b, ord=2)

# construct matrices with various eigenvalue distributions, but same condition number
As = []
eigs = []
cond = 0
for cluster_count in CLUSTER_COUNTS:
    cluster_size = PROBLEM_SIZE // cluster_count
    eig = np.array([])
    lower_bound = EIGV_SAMPLE_RANGE[0] + CLUSTER_SPREAD
    upper_bound = EIGV_SAMPLE_RANGE[1] - CLUSTER_SPREAD
    centers = np.linspace(lower_bound, upper_bound, cluster_count, endpoint=True)
    for center in centers:
        even_dist = np.linspace(
            center - CLUSTER_SPREAD, center + CLUSTER_SPREAD, cluster_size
        )
        eig = np.append(eig, even_dist)
    eig_sorted = np.sort(eig)
    As.append(np.diag(eig_sorted))
    eigs.append(eig_sorted)

# experiments
figratio = len(CLUSTER_COUNTS) / 2  # height/width
fig, axs = plt.subplots(
    nrows=len(CLUSTER_COUNTS),
    ncols=2,
    sharex=False,
    sharey=False,
    squeeze=False,
    figsize=(FIGWIDTH, FIGWIDTH * figratio),
)
figs_seperate = []
axs_seperate = []
for _ in range(len(CLUSTER_COUNTS)):
    _fig, _axs = plt.subplots(
        nrows=1,
        ncols=2,
        sharex=False,
        sharey=False,
        squeeze=False,
        figsize=(FIGWIDTH, FIGWIDTH * figratio / len(CLUSTER_COUNTS)),
    )
    figs_seperate.append(_fig)
    axs_seperate.append(_axs)
domain_range = DOMAIN[1] - DOMAIN[0]
codomain_range = CODOMAIN[1] - CODOMAIN[0]
iteration_upperbound = 0
for i, A in enumerate(As):
    # select axis
    ax = axs[i, 0]
    ax_sep = axs_seperate[i][0, 0]

    # solve exact solution (use diagonality of A)
    x_exact = b / eigs[i]

    # solve cg for each matrix
    custom_cg = CustomCG(A, b, x0)
    x, success = custom_cg.solve(
        save_residuals=True, x_exact=x_exact
    )

    # upperbound for the number of iterations
    iteration_upperbound = custom_cg.calculate_iteration_upperbound()

    # calculate residual polynomials
    cg_poly_x, cg_poly_r, cg_poly_e = custom_cg.cg_polynomial(RESOLUTION, domain=DOMAIN)
    iterations = len(cg_poly_r) - 1

    # plot final residual polynomial(s)
    label = r"$r_m$"
    ax.plot(cg_poly_x, cg_poly_r[-1], label=label)
    ax_sep.plot(cg_poly_x, cg_poly_r[-1], label=label, color=ax.lines[-1].get_color())
    for j in range(1, NUM_RESPOLY):
        index = -(j + 1)
        if abs(index) > len(cg_poly_r):
            continue
        label = r"$r_{m-" + f"{j}" + r"}$"
        ax.plot(cg_poly_x, cg_poly_r[index], label=label, zorder=9)
        ax_sep.plot(
            cg_poly_x,
            cg_poly_r[index],
            label=label,
            zorder=9,
            color=ax.lines[-1].get_color(),
        )

    # condition number of A
    cond = np.linalg.cond(A)

    # add eigenvalues of A
    ax.scatter(
        np.real(eigs[i]),
        np.imag(eigs[i]),
        marker=".",
        color="black",
        zorder=10,
        s=20,
    )
    ax_sep.scatter(
        np.real(eigs[i]),
        np.imag(eigs[i]),
        marker=".",
        color="black",
        zorder=10,
        s=20,
    )

    # axis properties
    ax.set_ylim(CODOMAIN)
    ax_sep.set_ylim(CODOMAIN)

    # number of iterations
    convergence_info = f"$m$ = {iterations}"

    # number of clusters
    n_c_text = r"$\mathbf{n_c = " + f"{CLUSTER_COUNTS[i]}" + "}$"

    ax.text(
        DOMAIN[0] + 0.5 * domain_range,
        CODOMAIN[0] + 0.9 * codomain_range,
        convergence_info,
        horizontalalignment="center",
        verticalalignment="top",
        zorder=11,
    )
    ax_sep.text(
        DOMAIN[0] + 0.5 * domain_range,
        CODOMAIN[0] + 0.9 * codomain_range,
        convergence_info,
        horizontalalignment="center",
        verticalalignment="top",
        zorder=11,
    )
    ax.text(
        DOMAIN[0] - 0.1 * domain_range,
        0,
        n_c_text,
        # rotation=270,
        verticalalignment="center",
        horizontalalignment="right",
    )
    ax_sep.text(
        DOMAIN[0] - 0.1 * domain_range,
        0,
        n_c_text,
        # rotation=270,
        verticalalignment="center",
        horizontalalignment="right",
    )

    # set classic axis style
    ax = mpl_graph_plot_style(
        ax, DOMAIN, CODOMAIN, origin=True, xtick_locs=[DOMAIN[1]], ytick_locs=CODOMAIN
    )
    ax_sep = mpl_graph_plot_style(
        ax_sep,
        DOMAIN,
        CODOMAIN,
        origin=True,
        xtick_locs=[DOMAIN[1]],
        ytick_locs=CODOMAIN,
    )

    # calculate resiudals
    ax = axs[i, 1]
    ax_sep = axs_seperate[i][0, 1]
    rel_residuals = custom_cg.get_relative_residuals()
    ax.semilogy(rel_residuals, label="residual  ratio")
    ax_sep.semilogy(rel_residuals, label="residual norm ratio")
    ax.set_xlabel(r"$\mathbf{m}$")
    ax_sep.set_xlabel(r"$\mathbf{m}$")
    ax.set_ylabel(r"$\mathbf{||r_m||_2/||r_0||_2}$")
    ax_sep.set_ylabel(r"$\mathbf{||r_m||_2/||r_0||_2}$")
    ax.set_ylim(bottom=1e-16, top=1e2)
    ax_sep.set_ylim(bottom=1e-16, top=1e2)
    ax.set_xlim(left=0, right=iteration_upperbound + 1)
    ax_sep.set_xlim(left=0, right=iteration_upperbound + 1)

    # plot convergence line
    ax.axhline(y=custom_cg.tol, linestyle="--", zorder=8)
    ax_sep.axhline(
        y=custom_cg.tol, color=ax.lines[-1].get_color(), linestyle="--", zorder=8
    )

    # print meta information
    print(
        f"#clusters: {CLUSTER_COUNTS[i]}, spread: {CLUSTER_SPREAD}, m_th: {iteration_upperbound}, ||r_m||/||r_0|| = {rel_residuals[-1]:.2e}"
    )


# plot legend last
pos_x = 0
pos_y = 0
height = 0.1 * codomain_range
width = 1.1 * domain_range
axs[0, 0].legend(
    fontsize=8,
    loc="right",
    ncol=3,
    mode="expand",
    bbox_to_anchor=(pos_x, pos_y, width, height),
).set_zorder(11)

# tight layout
fig.tight_layout()
for _fig in figs_seperate:
    _fig.tight_layout()

if ARGS.generate_output:
    save_latex_figure("cg_convergence_extreme_spectra", fig=fig)
    for i, _fig in enumerate(figs_seperate):
        save_latex_figure(f"cg_convergence_extreme_spectra_cluster{i}", fig=_fig)
if ARGS.show_output:
    fig.show()
    input("Press Enter to continue...")
    plt.close(fig)
    for _fig in figs_seperate:
        _fig.show()
        input("Press Enter to continue...")
        plt.close(_fig)
