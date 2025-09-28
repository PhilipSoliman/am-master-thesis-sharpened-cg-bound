import atexit
import os
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from bound_and_spectrum_vs_iterations_fig import SPECTRUM_PLOT_FREQ
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata
from sharpened_bound_vs_iterations import (
    N_ITERATIONS,
    PRECONDITIONERS,
    get_spectrum_save_path,
)

from hcmsfem import LOGGER
from hcmsfem.cli import get_cli_args
from hcmsfem.meshes import DefaultQuadMeshParams
from hcmsfem.plot_utils import CustomColors
from hcmsfem.preconditioners import (
    AMSCoarseSpace,
    GDSWCoarseSpace,
    RGDSWCoarseSpace,
    TwoLevelSchwarzPreconditioner,
)
from hcmsfem.problems import CoefFunc
from hcmsfem.root import get_venv_root
from hcmsfem.solvers import CGIterationBound

CLI_ARGS = get_cli_args()


# save directory for the table
SAVE_DIR = get_venv_root() / "tables"

MESHES = [
    DefaultQuadMeshParams.Nc4,
    DefaultQuadMeshParams.Nc16,
    DefaultQuadMeshParams.Nc64,
]
PRECONDITIONERS = [
    (TwoLevelSchwarzPreconditioner, AMSCoarseSpace),
    (TwoLevelSchwarzPreconditioner, GDSWCoarseSpace),
    (TwoLevelSchwarzPreconditioner, RGDSWCoarseSpace),
]
MAX_ITERS_PER_PRECONDITIONER = [100, 100, 300]

# define max iters per preconditioner iterable
class MaxItersPerPreconditioner:
    def __init__(self):
        self.max_iters_list = [f"{m} ({p[1].SHORT_NAME})" for m, p in zip(MAX_ITERS_PER_PRECONDITIONER, PRECONDITIONERS)]

    def __iter__(self):
        return iter(self.max_iters_list)

# colour gradient for the table
nan_color = "#f0f0f0"
warning_color = CustomColors.RED.value
middle_color = "#e2e4fb"
high_color = CustomColors.SKY.value
three_color = LinearSegmentedColormap.from_list(
    "threecolor", [warning_color, middle_color, high_color]
)


def generate_iteration_bound_table(
    coef_func: CoefFunc,
    max_iters: Optional[int] = None,
    max_iters_per_preconditioner: Optional[list[int]] = None,
    max_iter_percentage: float = 0.5,
    show: bool = False,
):
    """
    Generate a table of CG iteration bounds for a given coefficient function.

    Args:
        coef_func (CoefFunc): The coefficient function to use.
        max_iters (Optional[int]): Maximum number of iterations to consider for the table.
            If None, it will be determined based on the length of the eigenvalues.
        max_iter_percentage (float): Percentage of the iterations to convergence to consider.
            Defaults to 0.75.
    """

    # design table structure

    bounds = [
        "$m_1$",
        "$m_s$",
    ]
    cidx = pd.Index(["$m$"] + bounds + ["$i$"])
    meshes_names = [
        f"$\\mathbf{{H=1/{int(1 / mesh_params.coarse_mesh_size)}}}$"
        for mesh_params in MESHES
    ]
    coarse_space_names = [
        coarse_space_cls.SHORT_NAME for _, coarse_space_cls in PRECONDITIONERS
    ]
    iidx = pd.MultiIndex.from_product(
        [
            meshes_names,
            coarse_space_names,
        ]
    )

    # get data from saved arrays
    data = []
    differences = []
    for mesh_params in MESHES:
        for i, (preconditioner_cls, coarse_space_cls) in enumerate(PRECONDITIONERS):
            fp = get_spectrum_save_path(
                mesh_params, coef_func, preconditioner_cls, coarse_space_cls
            )

            if fp.exists():
                # load data from the saved numpy file
                array_zip = np.load(fp)
                m = len(array_zip["eigenvalues"])
                m_classic = array_zip["classic_bound"]
                m_multi_cluster = array_zip["multi_cluster_bound"]
                cluster_convergence_iterations = array_zip[
                    "cluster_convergence_iterations"
                ]

                # determine the maximum number of iterations
                _max_iters = min(N_ITERATIONS, round(m * max_iter_percentage))
                if isinstance(max_iters_per_preconditioner, list):
                    max_iters = max_iters_per_preconditioner[i]
                _max_iters = (
                    min(_max_iters, max_iters) if max_iters is not None else _max_iters
                )

                # get most recent iteration for the bounds
                if not np.any(cluster_convergence_iterations <= _max_iters):
                    LOGGER.warning(
                        "No convergence iterations found within the specified max_iters."
                    )
                    cluster_convergence_iterations = np.array([_max_iters])
                mask = cluster_convergence_iterations <= _max_iters
                iteration = (
                    cluster_convergence_iterations[mask][-1] if np.any(mask) else None
                )

                # get most recent bounds but limited to max_iters
                m_classic_b = m_classic[mask][-1]
                diff_m_classic = int(m_classic_b - m)

                m_multi_cluster_b = m_multi_cluster[mask][-1]
                diff_m_multi_cluster = int(m_multi_cluster_b - m)

                # construct row
                data.append(
                    [
                        m,
                        m_classic_b,
                        m_multi_cluster_b,
                        iteration,
                    ]
                )

                differences.append(
                    [
                        diff_m_classic,
                        diff_m_multi_cluster,
                    ]
                )

            else:
                # Provide a clickable link to the script in the repo using Rich markup with absolute path
                approx_path = Path(__file__).parent / "sharerations.py"
                LOGGER.error(
                    f"File %s does not exist. Run '[link=file:{approx_path}]sharpened_bound_vs_iterations.py[/link]' first.",
                    fp,
                )
                exit()

    # create DataFrame
    data = np.array(data).T
    df = pd.DataFrame(
        data,
        columns=iidx,
        index=cidx,
    )

    # get styler
    styler = df.style

    # set number precision and thousands format for all columns
    styler.format(precision=0, na_rep="-", thousands=",")

    # apply background gradient to the bound columns
    def unmark_nan(s):  # nan colour
        return [f"background-color: {nan_color}" if pd.isna(v) else "" for v in s]

    for i, (mesh_name, coarse_space_name) in enumerate(df.columns):
        subset = pd.IndexSlice[bounds, [(mesh_name, coarse_space_name)]]
        diff = np.array(differences[i])
        negative_diffs = np.abs(diff[diff < 0])
        positive_diffs = np.abs(diff[diff >= 0])

        # rank differences from smallest to largest, substract 1 for 0-based index
        negative_rank = rankdata(negative_diffs, method="dense") - 1
        positive_rank = rankdata(positive_diffs, method="dense") - 1

        # normalize scores (negative to 0 to 0.5 and positive from 0.5 to 1)
        negative_scores = (
            0.5 * (negative_rank / len(np.unique(negative_diffs)))
            if len(negative_diffs) > 0
            else np.array([])
        )

        positive_scores = (
            0.5 * (2 - positive_rank / len(np.unique(positive_diffs)))
            if len(positive_diffs) > 0
            else np.array([])
        )

        # apply gradient for possible lower bounds (warning)
        score = np.zeros(len(diff))
        score[diff < 0] = negative_scores
        score[diff >= 0] = positive_scores
        styler.background_gradient(
            cmap=three_color,
            subset=subset,
            axis=0,
            gmap=score,
            vmin=0,
            vmax=1,
        ).apply(subset=subset, axis=0, func=unmark_nan)

    # (Optional) set bold font for indices with plain text names (does not apply to LaTeX)
    # styler.map_index(lambda v: "font-weight: bold;", level=1, axis=0)

    # table file path
    Nc = int(1 / mesh_params.coarse_mesh_size)
    tab_fp = SAVE_DIR / f"dd29_cg_iteration_bound.tex"

    # table caption
    caption = (
        f"The number of PCG iterations required to achieve convergence $m$ and corresponding iteration bounds {', '.join(bounds)}. "
        r"Bounds are calculated with Ritz spectrum PCG at the $i^{\textrm{th}}$ iteration for subdomain sizes "
        f"{', '.join(meshes_names)} "
        f"and 2-OAS preconditioner with {', '.join(coarse_space_names)} coarse spaces. "
        "Cell colors indicate if bounds are larger (blue) or smaller (red) than $m$, with shading proportional to absolute difference. "
        f"Bounds are calculated with $\eta={SPECTRUM_PLOT_FREQ}$, $\\tau={CGIterationBound.CLUSTER_CONVERGENCE_TOLERANCE}$, $i_{{\max}}$={', '.join(MaxItersPerPreconditioner())} and $r={max_iter_percentage}$."
    )

    styler.to_latex(
        tab_fp,
        caption=caption,
        position="H",
        label=f"tab:cg_iteration_bounds",
        clines="all;data",
        convert_css=True,
        position_float="centering",
        multicol_align="c",
        hrules=True,
    )

    if show:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            styler.to_html(
                tmp.name,
                caption=caption,
                label=f"tab:cg_iteration_bound_Nc={Nc}_N={max_iters}",
                clines="skip-last;data",
                convert_css=True,
                position_float="centering",
                multicol_align="c",
                hrules=True,
            )

            webbrowser.open(f"file://{tmp.name}")

        # Register cleanup
        def cleanup_tmp():
            try:
                os.remove(tmp.name)
            except Exception:
                print(f"Could not delete temporary file {tmp.name}")

        atexit.register(cleanup_tmp)


max_iters = 300
max_iter_percentage = 0.5
if CLI_ARGS.generate_output:
    generate_iteration_bound_table(
        CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS,
        max_iters_per_preconditioner=MAX_ITERS_PER_PRECONDITIONER,
        max_iter_percentage=max_iter_percentage,
    )
elif CLI_ARGS.show_output:
    generate_iteration_bound_table(
        CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS,
        show=True,
        max_iters_per_preconditioner=MAX_ITERS_PER_PRECONDITIONER,
        max_iter_percentage=max_iter_percentage,
    )

    input("Press Enter to exit...")
