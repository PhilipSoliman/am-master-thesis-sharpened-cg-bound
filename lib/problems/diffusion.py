from enum import Enum

import ngsolve as ngs
import numpy as np

from lib.boundary_conditions import BoundaryConditions, HomogeneousDirichlet
from lib.meshes import TwoLevelMesh
from lib.preconditioners import AMSCoarseSpace, TwoLevelSchwarzPreconditioner
from lib.problem_type import ProblemType
from lib.problems.problem import Problem


class SourceFunc(Enum):
    """Enumeration for source functions used in the diffusion problem."""

    CONSTANT = "constant_source"
    PARABOLIC = "parabolic_source"


class CoefFunc(Enum):
    """Enumeration for coefficient functions used in the diffusion problem."""

    SINUSOIDAL = "sinusoidal_coefficient"
    INCLUSIONS = "inclusions_coefficient"
    INCLUSIONS_2LAYERS = "inclusions_2layers_coefficient"
    INCLUSIONS_EDGES = "inclusions_edges_coefficient"
    CONSTANT = "constant_coefficient"
    HETMANIUK_LEHOUCQ = "hetmaniuk_lehoucq_coefficient"
    HEINLEIN = "heinlein_coefficient"


class DiffusionProblem(Problem):
    """Class representing a diffusion problem on a two-level mesh."""

    def __init__(
        self,
        boundary_conditions: BoundaryConditions,
        lx=1.0,
        ly=1.0,
        coarse_mesh_size=0.15,
        refinement_levels=2,
        layers=2,
        coef_func=CoefFunc.INCLUSIONS,
        source_func=SourceFunc.CONSTANT,
    ):
        try:
            two_mesh = TwoLevelMesh.load(
                lx=lx,
                ly=ly,
                coarse_mesh_size=coarse_mesh_size,
                refinement_levels=refinement_levels,
                layers=layers,
            )
        except FileNotFoundError:
            print("Mesh file not found. Creating a new mesh.")
            two_mesh = TwoLevelMesh(lx, ly, coarse_mesh_size, refinement_levels, layers)

            print("Saving newly created mesh to file...")
            two_mesh.save()

        # initialize the Problem with the TwoLevelMesh
        ptype = ProblemType.DIFFUSION
        self.boundary_conditions = boundary_conditions
        super().__init__(two_mesh, [boundary_conditions], ptype)

        # construct finite element space
        self.construct_fespace()

        # get coefficient and source functions
        self.coef_func_name = coef_func.value
        self.coef_func = getattr(self, self.coef_func_name)()
        self.source_func_name = source_func.value
        self.source_func = getattr(self, self.source_func_name)()

    def assemble(self, gfuncs=None):
        return super().assemble(gfuncs=[self.coef_func, self.source_func])

    ####################
    # source functions #
    ####################
    def constant_source(self):
        """Constant source function."""
        return 1.0

    def parabolic_source(self):
        """Parabolic source function."""
        lx, ly = self.two_mesh.lx, self.two_mesh.ly
        return 32 * (ngs.y * (ly - ngs.y) + ngs.x * (lx - ngs.x))

    #########################
    # coefficient functions #
    #########################
    # eq. 5.5 from Hetmaniuk & Lehoucq (2010).
    def constant_coefficient(self):
        return 1.0

    # eq. 5.6 from Hetmaniuk & Lehoucq (2010).
    def hetmaniuk_lehoucq_coefficient(self):
        c = (
            1.2 + ngs.cos(32 * ngs.pi * ngs.x * (1 - ngs.x) * ngs.y * (1 - ngs.y))
        ) ** -1
        return c

    # eq. 5.8 from Hetmaniuk & Lehoucq (2010)
    def sinusoidal_coefficient(self):
        sx, sy = ngs.sin(25 * ngs.pi * ngs.x / self.two_mesh.lx), ngs.sin(
            25 * ngs.pi * ngs.y / self.two_mesh.ly
        )
        cy = ngs.cos(25 * ngs.pi * ngs.y / self.two_mesh.ly)
        c = ((2 + 1.8 * sx) / (2 + 1.8 * cy)) + ((2 + sy) / (2 + 1.8 * sx))
        return c

    # eq. 4.36 from Heinlein (2016).
    def heinlein_coefficient(self):
        sx, sy = ngs.sin(25 * ngs.pi * ngs.x), ngs.sin(25 * ngs.pi * ngs.y)
        cy = ngs.cos(25 * ngs.pi * ngs.y)
        c = ((2 + 1.99 * sx) / (2 + 1.99 * cy)) + ((2 + sy) / (2 + 1.99 * sx))
        return c

    # High coefficient inclusions around the coarse nodes.
    def inclusions_coefficient(self):
        # setup pieewise constant coefficient function
        constant_fes = ngs.L2(self.two_mesh.fine_mesh, order=0)
        grid_func = ngs.GridFunction(constant_fes)

        # get mesh size
        H = self.two_mesh.coarse_mesh_size
        h = H * 2 ** (-self.two_mesh.refinement_levels)

        # define background and contrast values
        background = 1.0
        contrast = 1e8

        # Get all free coarse node coordinates
        free_coarse_nodes = list(self.fes.free_component_tree_dofs.keys())
        coarse_points = [
            self.two_mesh.coarse_mesh[node].point for node in free_coarse_nodes
        ]

        # Loop over coarse nodes and set high contrast if inside any inclusion
        coef_array = np.full(self.two_mesh.fine_mesh.ne, background)
        for coarse_node, coarse_point in zip(free_coarse_nodes, coarse_points):
            mesh_el = self.two_mesh.fine_mesh[coarse_node]
            for el in mesh_el.elements:
                vertices = np.array(
                    [
                        self.two_mesh.fine_mesh[v].point
                        for v in self.two_mesh.fine_mesh[el].vertices
                    ]
                )
                d = np.linalg.norm(vertices - coarse_point, axis=1, ord=2)
                if np.all(d < 2 * h):
                    coef_array[el.nr] = contrast

        grid_func.vec.FV().NumPy()[:] = coef_array
        coef_func = ngs.CoefficientFunction(grid_func)
        return coef_func.Compile()

    # High coefficient inclusions on the coarse node (2 layers).
    def inclusions_2layers_coefficient(self):
        # setup piecewise constant coefficient function
        constant_fes = ngs.L2(self.two_mesh.fine_mesh, order=0)
        grid_func = ngs.GridFunction(constant_fes)

        # define background and contrast values
        background = 1.0
        contrast = 1e8

        # Get all free coarse node coordinates
        free_coarse_nodes = list(self.fes.free_component_tree_dofs.keys())

        # Loop over coarse nodes and set high contrast if inside any inclusion
        coef_array = np.full(self.two_mesh.fine_mesh.ne, background)
        for coarse_node in free_coarse_nodes:
            mesh_el = self.two_mesh.fine_mesh[coarse_node]
            elements = set(mesh_el.elements)
            for el in elements:
                coef_array[el.nr] = contrast
                # add second layer
                outer_vertices = set(self.two_mesh.fine_mesh[el].vertices) - set(
                    [coarse_node]
                )
                for vertex in outer_vertices:
                    outer_elements = (
                        set(self.two_mesh.fine_mesh[vertex].elements) - elements
                    )
                    for outer_el in outer_elements:
                        coef_array[outer_el.nr] = contrast

        grid_func.vec.FV().NumPy()[:] = coef_array
        coef_func = ngs.CoefficientFunction(grid_func)
        return coef_func.Compile()

    # edge inclusions
    def inclusions_edges_coefficient(self):
        # get fine mesh size
        num_edge_vertices = (
            2**self.two_mesh.refinement_levels - 2
        )  # without coarse nodes
        edge_vertex_inclusion_idxs = np.array(
            [i for i in range(2, num_edge_vertices, 2)]
        )

        # setup piecewise constant coefficient function
        constant_fes = ngs.L2(self.two_mesh.fine_mesh, order=0)
        grid_func = ngs.GridFunction(constant_fes)

        # define background and contrast values
        background = 1.0
        contrast = 1e8

        # asssemble grid function
        coef_array = np.full(self.two_mesh.fine_mesh.ne, background)
        coarse_edges = set(self.two_mesh.coarse_mesh.edges)
        component_tree_dofs = self.two_mesh.connected_component_tree
        free_coarse_nodes = list(self.fes.free_component_tree_dofs.keys())
        for coarse_node in free_coarse_nodes:
            node_data = component_tree_dofs[coarse_node]
            coarse_point = np.array(self.two_mesh.coarse_mesh[coarse_node].point)
            for coarse_edge, edge_data in node_data.items():
                if coarse_edge not in coarse_edges:
                    continue  # edge already processed
                coarse_edges.remove(coarse_edge)
                distances = []
                for vertex in edge_data["fine_vertices"]:
                    point = np.array(self.two_mesh.fine_mesh[vertex].point)
                    d = np.linalg.norm(coarse_point - point, ord=2)
                    distances.append((vertex, d))

                # Sort the (vertex, distance) pairs by distance
                sorted_vertices = [
                    v for v, d in sorted(distances, key=lambda pair: pair[1])
                ]
                for i in edge_vertex_inclusion_idxs:
                    vertex = sorted_vertices[i]
                    mesh_el = self.two_mesh.fine_mesh[vertex]
                    for el in mesh_el.elements:
                        coef_array[el.nr] = contrast

        grid_func.vec.FV().NumPy()[:] = coef_array
        coef_func = ngs.CoefficientFunction(grid_func)
        return coef_func.Compile()

    def save_functions(self):
        """Save the source and coefficient functions to vtk."""
        self.save_ngs_functions(
            funcs=[self.source_func, self.coef_func, self.u],
            names=[self.source_func_name, self.coef_func_name, "solution"],
            category="diffusion",
        )


def main():
    # Define mesh parameters
    lx = 1.0
    ly = 1.0
    coarse_mesh_size = 1 / 32
    refinement_levels = 4
    layers = 2

    # define source and coefficient functions
    source_func = SourceFunc.CONSTANT
    coef_func = CoefFunc.INCLUSIONS

    # create diffusion problem
    diffusion_problem = DiffusionProblem(
        HomogeneousDirichlet(ProblemType.DIFFUSION),
        lx=lx,
        ly=ly,
        coarse_mesh_size=coarse_mesh_size,
        refinement_levels=refinement_levels,
        layers=layers,
        source_func=source_func,
        coef_func=coef_func,
    )
    print(diffusion_problem.boundary_conditions)

    # solve problem
    get_cg_info = True
    diffusion_problem.solve(
        preconditioner=TwoLevelSchwarzPreconditioner,
        coarse_space=AMSCoarseSpace,
        rtol=1e-8,
        save_cg_info=get_cg_info,
        save_coarse_bases=False,
    )

    # Save the functions to vtk files
    # diffusion_problem.save_functions()

    if get_cg_info:
        import matplotlib.pyplot as plt
        import numpy as np

        from lib.utils import set_mpl_cycler, set_mpl_style

        set_mpl_style()
        set_mpl_cycler(colors=True, lines=True)

        fig, axs = plt.subplots(
            2,
            2,
            figsize=(10, 6),
            gridspec_kw={"height_ratios": [3, 1], "width_ratios": [1, 1]},
        )

        # Remove the bottom-right axis and make the bottom-left axis span both columns
        fig.delaxes(axs[1, 0])
        fig.delaxes(axs[1, 1])
        gs = axs[1, 0].get_gridspec()
        axs_bottom = fig.add_subplot(gs[1, :])

        # plot the coefficients
        axs[0, 0].plot(diffusion_problem.cg_alpha, label=r"$\alpha$")
        axs[0, 0].plot(diffusion_problem.cg_beta, label=r"$\beta$")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Coefficient Value")
        axs[0, 0].legend()

        # plot residuals and preconditioned residuals
        axs[0, 1].plot(diffusion_problem.cg_residuals, label=r"$||r_m||_2 / ||r_0||_2$")
        if diffusion_problem.cg_precond_residuals is not None:
            axs[0, 1].plot(
                diffusion_problem.cg_precond_residuals, label=r"$||z_m||_2 / ||z_0||_2$"
            )
        axs[0, 1].set_xlabel("Iteration")
        axs[0, 1].set_ylabel("Relative residuals")
        axs[0, 1].set_yscale("log")
        axs[0, 1].legend()

        plt.suptitle(
            f"CG convergence ("
            + r"$\mathcal{C}$"
            + f" = {coef_func.name}, "
            + r"$f$"
            + f" = {source_func.name}, "
            + r"$M^{-1}$"
            + f" = {diffusion_problem.precond_name})"
        )

        # Plot each spectrum at a different y
        axs_bottom.plot(
            np.real(diffusion_problem.approximate_eigs),
            np.full_like(diffusion_problem.approximate_eigs, 0),
            marker="x",
            linestyle="None",
        )

        # Set y-ticks and labels
        axs_bottom.set_ylim(-0.5, 0.5)
        axs_bottom.set_yticks([0], ["$\\mathbf{\\sigma(T_m)}$"])
        axs_bottom.set_xscale("log")
        axs_bottom.grid(axis="x")
        axs_bottom.grid()
        ax2 = axs_bottom.twinx()

        # add condition numbers on right axis
        def format_cond(c):
            if np.isnan(c):
                return "n/a"
            mantissa, exp = f"{c:.1e}".split("e")
            exp = int(exp)
            return rf"${mantissa} \times 10^{{{exp}}}$"

        cond = np.max(diffusion_problem.approximate_eigs) / np.min(
            diffusion_problem.approximate_eigs
        )
        ax2.set_ylim(axs_bottom.get_ylim())
        ax2.set_yticks([0], [format_cond(cond)])

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
