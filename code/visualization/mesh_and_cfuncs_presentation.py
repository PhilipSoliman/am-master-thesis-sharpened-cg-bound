from mesh_and_coefficient_functions_fig import *
from hcmsfem.root import get_venv_root

IMAGE_DIR = get_venv_root() / "images"

def reformat_fig(fig: plt.Figure) -> plt.Figure:
    fig.patch.set_facecolor(CustomColors.NAVY.value)
    # set font colors to white
    for ax in fig.axes:
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.set_facecolor(CustomColors.NAVY.value)
        ax.title.set_color("white")

    return fig

def plot_meshes_and_domains_png(two_mesh: TwoLevelMesh) -> plt.Figure:
    fig = plot_meshes_and_domains(two_mesh)
    return reformat_fig(fig)

def plot_coefficient_functions_png(two_mesh: TwoLevelMesh) -> plt.Figure:
    fig = plot_coefficient_functions(two_mesh)
    return reformat_fig(fig)

if __name__ == "__main__":
    two_mesh_4 = TwoLevelMesh(mesh_params=DefaultQuadMeshParams.Nc4)
    figs = [
        plot_meshes_and_domains_png(two_mesh_4),
        plot_coefficient_functions_png(two_mesh_4),
    ]
    fns = [
        "meshes_and_domains",
        "coefficient_functions",
    ]
    for fig, fn in zip(figs, fns):
        fig.tight_layout()
        if CLI_ARGS.generate_output:
            fig.savefig(IMAGE_DIR / f"{fn}.png", dpi=1000)
    if CLI_ARGS.show_output:
        plt.show()
