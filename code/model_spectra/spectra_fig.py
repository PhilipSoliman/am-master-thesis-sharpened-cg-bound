from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from approximate_spectra import (
    COEF_FUNCS,
    MESHES,
    PRECONDITIONERS,
    get_spectrum_save_path,
)

from hcmsfem.cli import get_cli_args
from hcmsfem.logger import LOGGER
from hcmsfem.meshes import DefaultQuadMeshParams
from hcmsfem.plot_utils import save_latex_figure, set_mpl_cycler, set_mpl_style
from hcmsfem.problems import CoefFunc
from hcmsfem.solvers import partition_eigenspectrum, partition_eigenspectrum_tails

# set matplotlib style & cycler
set_mpl_style()
set_mpl_cycler(colors=True)

# get cli args
ARGS = get_cli_args()
FIGWIDTH = 3
FIGHEIGHT = 2

MESHES = [DefaultQuadMeshParams.Nc4, DefaultQuadMeshParams.Nc64]
COEF_FUNCS = [
    CoefFunc.THREE_LAYER_VERTEX_INCLUSIONS,
    CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS,
]
FONTSIZE = 9
MARKERSIZE = 4


def plot_spectra(tail: bool) -> plt.Figure:
    # initialize figure and axes
    fig, axs = plt.subplots(
        len(COEF_FUNCS),
        len(MESHES),
        figsize=(FIGWIDTH * len(MESHES), FIGHEIGHT * len(COEF_FUNCS)),
        squeeze=False,
        # sharex=True,
        sharey=True,
    )

    # main plot loop
    for i, mesh_params in enumerate(MESHES):
        axes = axs[:, i]
        for coef_func, ax in zip(COEF_FUNCS, axes):
            spectra = {}
            split_indices = []
            tail_eigenvalues = []
            tail_indices = []
            cond_numbers = []
            niters = []
            global_min = None
            global_max = None
            for preconditioner_cls, coarse_space_cls in PRECONDITIONERS:
                # Load the eigenvalues from the saved numpy file
                fp = get_spectrum_save_path(
                    mesh_params, coef_func, preconditioner_cls, coarse_space_cls
                )
                if fp.exists():
                    eigenvalues = np.load(fp)["eigenvalues"]
                    # shorthand = (
                    #     f"{preconditioner_cls.SHORT_NAME}-{coarse_space_cls.SHORT_NAME}"
                    # )
                    shorthand  = f"{coarse_space_cls.SHORT_NAME}"

                    # store the eigenvalues in the spectra dictionary
                    # spectra[f"{shorthand:<12}"] = eigenvalues
                    spectra[shorthand] = eigenvalues

                    # store the split indices for the sharpened bound (except for the last one)
                    tail_eigs = []
                    tail_indxs = []
                    if tail:  # tail-cluster partitioning
                        split_indices.append(
                            partition_eigenspectrum_tails(
                                eigenvalues,
                                log_rtol=np.log(1e-8),
                                tail_eigenvalues=tail_eigs,
                                tail_indices=tail_indxs,
                            )
                        )
                    else:  # multi-cluster partitioning
                        split_idxs = partition_eigenspectrum(eigenvalues)
                        start = 0
                        for end in split_idxs:
                            if start == end:
                                tail_eigs.append(eigenvalues[start])
                                tail_indxs.append(start)
                        split_indices.append(split_idxs)
                    tail_eigenvalues.append(tail_eigs)
                    tail_indices.append(tail_indxs)

                    # get cluster coordinates
                    min_eig = np.min(np.abs(eigenvalues))
                    max_eig = np.max(np.abs(eigenvalues))

                    # calculate condition number
                    cond_numbers.append(
                        np.abs(max_eig / min_eig)
                        if len(eigenvalues) > 0 and min_eig > 0
                        else np.nan
                    )

                    # store the number of iterations
                    niters.append(len(eigenvalues))

                    # update global min and max (used for plotting number of iterations)
                    if global_min is None or min_eig < global_min:
                        global_min = min_eig
                    if global_max is None or max_eig > global_max:
                        global_max = max_eig
                else:
                    # Provide a clickable link to the script in the repo using Rich markup with absolute path
                    approx_path = Path(__file__).parent / "approximate_spectra.py"
                    LOGGER.error(
                        f"File %s does not exist. Run '[link=file:{approx_path}]approximate_spectra.py[/link]' first.",
                        fp,
                    )
                    exit()

            # spectra
            for idx, eigenvalues in enumerate(spectra.values()):
                # seperate cluster and tail eigenvalues
                sep_eigenvalues = eigenvalues[~np.isin(eigenvalues, tail_eigenvalues[idx])]

                # eigenvalues
                line = ax.plot(
                    np.real(sep_eigenvalues),
                    np.full_like(sep_eigenvalues, idx),
                    marker="x",
                    linestyle="None",
                    zorder=5,
                    markersize=MARKERSIZE,
                )

                # tail eigenvalues
                ax.plot(
                    np.real(tail_eigenvalues[idx]),
                    np.full_like(tail_eigenvalues[idx], idx),
                    marker="o",
                    linestyle="None",
                    zorder=5,
                    color=line[0].get_color(),
                    markersize=MARKERSIZE
                )

                # tail cluster indices
                ax.plot(
                    np.real(eigenvalues[tail_indices[idx]]),
                    np.full_like(tail_indices[idx], idx),
                    marker="|",
                    linestyle="None",
                    color="black",
                    zorder=5,
                    markersize=10,
                )

                # split indices (plot identified clusters)
                fontsize = 15
                ax.text(
                    np.real(eigenvalues[0]),
                    idx,
                    "[",
                    color="black",
                    fontsize=fontsize,
                    va="center",
                    ha="right",
                    zorder=10,
                )
                for s_idx in split_indices[idx]:
                    ax.text(
                        np.real(eigenvalues[s_idx]),
                        idx,
                        "]",
                        color="black",
                        fontsize=fontsize,
                        va="center",
                        ha="left",
                        zorder=10,
                    )
                    if len(eigenvalues) > s_idx + 1:
                        ax.text(
                            np.real(eigenvalues[s_idx + 1]),
                            idx,
                            "[",
                            color="black",
                            fontsize=fontsize,
                            va="center",
                            ha="right",
                            zorder=10,
                        )
            ax.set_xscale("log")

            # preconditioner names
            ax.set_yticks(
                range(len(spectra)), list(spectra.keys()), fontsize=FONTSIZE, fontweight="bold", rotation=45
            )
            ax.set_ylim(-0.5, len(spectra) - 1 + 0.5)

            # condition numbers
            ax2 = ax.twinx()

            def format_cond(c):
                if np.isnan(c):
                    return "n/a"
                mantissa, exp = f"{c:.1e}".split("e")
                exp = int(exp)
                return rf"${mantissa} \times 10^{{{exp}}}$"

            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(range(len(cond_numbers)))
            ax2.set_yticklabels(
                [format_cond(c) if not np.isnan(c) else "n/a" for c in cond_numbers],
                fontsize=FONTSIZE, rotation=-45
            )

            # iteration counts
            x_niter = 10 ** (0.5 * (np.log10(global_min) + np.log10(global_max)))
            for idx, niter in enumerate(niters):
                y_niter = idx + 0.25
                ax.text(
                    x_niter,
                    y_niter,
                    rf"$m = {niter}$",
                    ha="center",
                    va="center",
                    fontsize=FONTSIZE,
                    color="black",
                    clip_on=False,
                )

            # grid settings
            ax.grid(axis="x", which="both", linestyle="--", linewidth=0.7)
            ax.grid(axis="y", which="both", linestyle=":", linewidth=0.5)

    # Add column titles (LaTeX, Nc as integer, no bold for compatibility)
    for col_idx, mesh_params in enumerate(MESHES):
        H = mesh_params.coarse_mesh_size
        Nc = int(1 / H)
        ax = axs[0, col_idx] if hasattr(axs, "ndim") and axs.ndim == 2 else axs[0]
        ax.set_title(rf"$\mathbf{{H = 1/{Nc}}}$", fontsize=FONTSIZE+1)

    # Add row labels (rotated, bold, fontsize 9) at the beginning of each row
    for row_idx, coef_func in enumerate(COEF_FUNCS):
        # Get the y-position as the center of the row of axes
        ax = axs[row_idx, 0]

        # Use axes coordinates to place the text just outside the left of the axes
        fig.text(
            0,  # x-position (fraction of figure width, adjust as needed)
            ax.get_position().y0
            + ax.get_position().height / 2,  # y-position (center of the row)
            coef_func.latex,  # use LaTeX representation
            va="center",
            ha="left",
            rotation=90,
            fontweight="bold",
            fontsize=FONTSIZE+2,
        )

    # tight layout for the figure
    fig.tight_layout(pad=1.3)

    return fig


if __name__ == "__main__":
    # plot the spectra
    figs = [plot_spectra(tail=False), plot_spectra(tail=True)]
    fp = Path(__file__).name.replace("_fig.py", "")
    fns = ["multi_cluster", "tail_cluster"]

    # save the figure if requested
    if ARGS.generate_output:
        for fig, fn in zip(figs, fns):
            save_latex_figure(f"{fp}_{fn}", fig)

    # show the figure if requested
    if ARGS.show_output:
        plt.show()
