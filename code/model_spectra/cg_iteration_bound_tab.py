import tempfile
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sharpened_bound_vs_iterations import (
    MESHES,
    N_ITERATIONS,
    PRECONDITIONERS,
    get_spectrum_save_path,
)

from hcmsfem import LOGGER
from hcmsfem.cli import CLI_ARGS
from hcmsfem.problems import CoefFunc
from hcmsfem.root import get_venv_root

# save directory for the table
SAVE_DIR = get_venv_root() / "tables"


def generate_iteration_bound_table(
    coef_func: CoefFunc,
    max_iters: Optional[int] = None,
    max_iter_percentage: float = 0.75,
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
        "$m_{N_{\\text{cluster}}}$",
        "$m_{N_{\\text{tail-cluster}}}$",
        "$m_{\\text{estimate}}$",
    ]
    bound_and_iter = ["bound", "iter."]
    cidx = pd.MultiIndex.from_product([bounds, bound_and_iter])
    cidx = pd.MultiIndex.from_arrays(
        [
            ["$m$"] + cidx.get_level_values(0).tolist(),
            [""] + cidx.get_level_values(1).tolist(),
        ]
    )
    meshes_names = [
        f"$H=1/{int(1 / mesh_params.coarse_mesh_size)}$" for mesh_params in MESHES
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
        for preconditioner_cls, coarse_space_cls in PRECONDITIONERS:
            fp = get_spectrum_save_path(
                mesh_params, coef_func, preconditioner_cls, coarse_space_cls
            )

            if fp.exists():
                # load data from the saved numpy file
                array_zip = np.load(fp)
                m = len(array_zip["eigenvalues"])
                m_classic = array_zip["classic_bound"]
                m_multi_cluster = array_zip["multi_cluster_bound"]
                m_tail_cluster = array_zip["tail_cluster_bound"]
                m_estimate = array_zip["estimate"]

                # determine the maximum number of iterations
                _max_iters = (
                    min(N_ITERATIONS, round(m * max_iter_percentage))
                    if max_iters is None
                    else max_iters
                )

                # get most recent bounds but limited to max_iters
                mask = (
                    m_classic[:, 0] <= _max_iters
                    if m_classic.size
                    else np.array([], dtype=bool)
                )
                m_classic_i, m_classic_b = (
                    m_classic[mask][-1] if np.any(mask) else (None, None)
                )
                diff_m_classic = (
                    abs(m_classic_b - m) if m_classic_b is not None else np.nan
                )

                mask = (
                    m_multi_cluster[:, 0] <= _max_iters
                    if m_multi_cluster.size
                    else np.array([], dtype=bool)
                )
                m_multi_cluster_i, m_multi_cluster_b = (
                    m_multi_cluster[mask][-1] if np.any(mask) else (None, None)
                )
                diff_m_multi_cluster = (
                    abs(m_multi_cluster_b - m)
                    if m_multi_cluster_b is not None
                    else np.nan
                )

                mask = (
                    m_tail_cluster[:, 0] <= _max_iters
                    if m_tail_cluster.size
                    else np.array([], dtype=bool)
                )
                m_tail_cluster_i, m_tail_cluster_b = (
                    m_tail_cluster[mask][-1] if np.any(mask) else (None, None)
                )
                diff_m_tail_cluster = (
                    abs(m_tail_cluster_b - m)
                    if m_tail_cluster_b is not None
                    else np.nan
                )

                mask = (
                    m_estimate[:, 0] <= _max_iters
                    if m_estimate.size
                    else np.array([], dtype=bool)
                )
                m_estimate_i, m_estimate_b = (
                    m_estimate[mask][-1] if np.any(mask) else (None, None)
                )
                diff_m_estimate = (
                    abs(m_estimate_b - m) if m_estimate_b is not None else np.nan
                )

                # construct row
                data.append(
                    [
                        m,
                        m_classic_b,
                        m_classic_i,
                        m_multi_cluster_b,
                        m_multi_cluster_i,
                        m_tail_cluster_b,
                        m_tail_cluster_i,
                        m_estimate_b,
                        m_estimate_i,
                    ]
                )

                differences.append(
                    [
                        diff_m_classic,
                        diff_m_multi_cluster,
                        diff_m_tail_cluster,
                        diff_m_estimate,
                    ]
                )

            else:
                # Provide a clickable link to the script in the repo using Rich markup with absolute path
                approx_path = Path(__file__).parent / "sharpened_bound_vs_iterations.py"
                LOGGER.error(
                    f"File %s does not exist. Run '[link=file:{approx_path}]sharpened_bound_vs_iterations.py[/link]' first.",
                    fp,
                )
                exit()

    # create DataFrame
    df = pd.DataFrame(
        data,
        columns=cidx,
        index=iidx,
    )

    # get styler
    styler = df.style

    # set precision for all columns
    styler.format(precision=0, na_rep="-", thousands=",")

    # TODO: def bound_color function in terms of rel. difference with actual number of iterations
    def unmark_nan(s):
        return ["background-color: #add8e6" if pd.isna(v) else "" for v in s]
    for i, idx in enumerate(df.index):
        subset = pd.IndexSlice[[idx], pd.IndexSlice[:, "bound"]]
        diff = differences[i]
        diff = np.argsort(diff)  # sort indices of differences
        styler.background_gradient(
            cmap="YlGn",
            subset=subset,
            axis=1,
            gmap= -diff,
        ).apply(subset=subset, func=unmark_nan)

    # rotate bound and iter column headers
    # styler.map_index(
    #     lambda v: "rotatebox:{45}--rwrap--latex;font-weight: bold;", level=1, axis=1
    # )
    styler.map_index(lambda v: "font-weight: bold;", level=0, axis=0)
    styler.map_index(lambda v: "font-weight: bold;", level=0, axis=1)

    # table file path
    Nc = int(1 / mesh_params.coarse_mesh_size)
    tab_fp = SAVE_DIR / f"cg_iteration_bound_coef={coef_func.short_name}.tex"

    # table caption
    caption = (
        f"PCG iteration bounds for solving the model diffusion problem with coefficient function {coef_func.latex}. Bounds are based on approximate spectra (Ritz values) obtained during the initial PCG iterations and are show for meshes "
        f"{', '.join(meshes_names)} "
        f"and 2-OAS preconditioners with {', '.join(coarse_space_names)} coarse spaces."
        " The $\\textbf{bound}$ columns show the values of the CG iteration bounds "
        f"{', '.join(bounds)}"
        " and the $\\textbf{iter.}$ columns show the iteration at which those bounds are obtained."
    )

    styler.to_latex(
        tab_fp,
        caption=caption,
        position="H",
        label=f"tab:cg_iteration_bound_coef={coef_func.short_name}",
        clines="all;data",
        convert_css=True,
        position_float="centering",
        multicol_align="|c|",
        hrules=True,
    )

    if show:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            styler.to_html(
                tmp.name,
                caption=caption,
                label=f"tab:cg_iteration_bound_Nc{Nc}_N={max_iters}",
                clines="skip-last;data",
                convert_css=True,
                position_float="centering",
                multicol_align="|c|",
                hrules=True,
            )

            webbrowser.open(f"file://{tmp.name}")


if CLI_ARGS.generate_output:
    generate_iteration_bound_table(CoefFunc.CONSTANT)
    generate_iteration_bound_table(CoefFunc.THREE_LAYER_VERTEX_INCLUSIONS)
    generate_iteration_bound_table(
        CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS, max_iters=300
    )
elif CLI_ARGS.show_output:
    generate_iteration_bound_table(CoefFunc.CONSTANT, show=True)
    generate_iteration_bound_table(
        CoefFunc.EDGE_SLABS_AROUND_VERTICES_INCLUSIONS, show=True
    )
    generate_iteration_bound_table(
        CoefFunc.THREE_LAYER_VERTEX_INCLUSIONS, show=True, max_iters=300
    )

generate_iteration_bound_table(CoefFunc.THREE_LAYER_VERTEX_INCLUSIONS, show=True)
