import matplotlib.pyplot as plt
import numpy as np

# from clibs.custom_cg import CustomCG
from lib import CustomCG
from lib.utils import (
    get_cli_args,
    mpl_graph_plot_style,
    save_latex_figure,
    set_mpl_cycler,
    set_mpl_style,
)

# constants
ARGS = get_cli_args()
FIGWIDTH = 5
LEGEND_HEIGHT = 0.1
DOMAIN = (0, 20)
CODOMAIN = (-1, 1)
ZOOMFIT = True
RESOLUTION = 1000
NUMMARKERS = 10

# set matplotlib style
set_mpl_style()
set_mpl_cycler(lines=False, markers=True, colors=True)

# create a random SPD matrix
n = 10
A = np.random.rand(n, n)
A = 0.5 * (A + A.T)
A += n * np.eye(n)

b = np.random.rand(n)
x0 = np.zeros(n)

# calculate exact solution for error calculation
x_exact = np.linalg.solve(A, b)

# create custom cg object
custom_cg = CustomCG(A, b, x0)

# solve the system
x, success = custom_cg.solve(save_residuals=True, x_exact=x_exact)

# calculate rel errors
rel_errors = custom_cg.get_relative_errors()

# calculate residual polynomials
domain_size = DOMAIN[1] - DOMAIN[0]
codomain_size = CODOMAIN[1] - CODOMAIN[0]
figratio = (1 + LEGEND_HEIGHT) * codomain_size / domain_size
cg_poly_x, cg_poly_r, cg_poly_e = custom_cg.cg_polynomial(
    RESOLUTION, domain=DOMAIN, respoly_error=True
)
figheight = FIGWIDTH if ZOOMFIT else FIGWIDTH * figratio
fig, ax = plt.subplots(1, 1, figsize=(FIGWIDTH, figheight))
x_ticks = np.linspace(*DOMAIN, 6)[1:]
y_ticks = np.linspace(*CODOMAIN, 6)
mpl_graph_plot_style(
    ax,
    domain=DOMAIN,
    codomain=CODOMAIN,
    origin=True,
    xtick_locs=x_ticks,
    ytick_locs=y_ticks,
)
print(ax)
for i in range(len(cg_poly_r)):
    if i > 0:
        label = r"$r_{" + str(i) + r"}(t)$"
        ax.plot(
            cg_poly_x, cg_poly_r[i], label=label, markevery=RESOLUTION // NUMMARKERS
        )
    print(
        f"CG iteration {i}: respoly error = {cg_poly_e[i]}, rel_error = {rel_errors[i]}"
    )

# add eigenvalues of A
ax.scatter(
    np.real(custom_cg.eigenvalues),
    np.imag(custom_cg.eigenvalues),
    marker=".",
    color="black",
    zorder=20,
)
ax.set_xlabel("t")
ax.set_ylabel("r(t)")
legend_height = (CODOMAIN[1] - CODOMAIN[0]) / 10  # 10% of the CODOMAIN
ax.legend(
    fontsize=8,
    loc="lower left",
    ncol=len(cg_poly_r) - 1,
    mode="expand",
    bbox_to_anchor=(0, -0.1, 1, 0.1),
)
plt.tight_layout()
if ARGS.generate_output:
    save_latex_figure("cg_convergence_behaviour")
if ARGS.show_output:
    plt.show()
