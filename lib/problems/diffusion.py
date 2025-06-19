from enum import Enum

import ngsolve as ngs
import numpy as np

from lib.boundary_conditions import BoundaryConditions, HomogeneousDirichlet
from lib.meshes import TwoLevelMesh
from lib.preconditioners import RGDSWCoarseSpace, TwoLevelSchwarzPreconditioner
from lib.problem_type import ProblemType
from lib.problems.problem import Problem


class SourceFunc(Enum):
    """Enumeration for source functions used in the diffusion problem."""

    CONSTANT = "constant_source"
    PARABOLIC = "parabolic_source"


class CoefFunc(Enum):
    """Enumeration for coefficient functions used in the diffusion problem."""

    SINUSOIDAL = "sinusoidal_coefficient"
    VERTEX_INCLUSIONS = "vertex_inclusions_coefficient"
    VERTEX_INCLUSIONS_2LAYERS = "vertex_inclusions_2layers_coefficient"
    EDGE_INCLUSIONS = "edge_inclusions_coefficient"
    EDGE_SLAB_INCLUSIONS = "edge_slab_inclusions_coefficient"
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
        coef_func=CoefFunc.VERTEX_INCLUSIONS,
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
        print("Assembling system")
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
    def constant_coefficient(self):
        """Constant coefficient function."""
        return 1.0

    def hetmaniuk_lehoucq_coefficient(self):
        """Hetmaniuk & Lehoucq coefficient function. Equation 5.6 from Hetmaniuk & Lehoucq (2010)."""
        c = (
            1.2 + ngs.cos(32 * ngs.pi * ngs.x * (1 - ngs.x) * ngs.y * (1 - ngs.y))
        ) ** -1
        return c

    def sinusoidal_coefficient(self):
        """Sinusoidal coefficient function. Equation 5.8 from Hetmaniuk & Lehoucq (2010)."""
        sx, sy = ngs.sin(25 * ngs.pi * ngs.x / self.two_mesh.lx), ngs.sin(
            25 * ngs.pi * ngs.y / self.two_mesh.ly
        )
        cy = ngs.cos(25 * ngs.pi * ngs.y / self.two_mesh.ly)
        c = ((2 + 1.8 * sx) / (2 + 1.8 * cy)) + ((2 + sy) / (2 + 1.8 * sx))
        return c

    def heinlein_coefficient(self):
        """Heinlein coefficient function. Equation 4.36 from Heinlein (2016)."""
        sx, sy = ngs.sin(25 * ngs.pi * ngs.x), ngs.sin(25 * ngs.pi * ngs.y)
        cy = ngs.cos(25 * ngs.pi * ngs.y)
        c = ((2 + 1.99 * sx) / (2 + 1.99 * cy)) + ((2 + sy) / (2 + 1.99 * sx))
        return c

    def vertex_inclusions_coefficient(self):
        """High coefficient inclusions around the coarse nodes."""
        # setup piecewise constant coefficient function
        constant_fes = ngs.L2(self.two_mesh.fine_mesh, order=0)
        grid_func = ngs.GridFunction(constant_fes)

        # define background and contrast values
        background = 1.0
        contrast = 1e8

        # Loop over coarse nodes and set high contrast on elements around them
        coef_array = np.full(self.two_mesh.fine_mesh.ne, background)
        for coarse_node in list(self.fes.free_component_tree_dofs.keys()):
            mesh_el = self.two_mesh.fine_mesh[coarse_node]
            for el in mesh_el.elements:
                coef_array[el.nr] = contrast

        grid_func.vec.FV().NumPy()[:] = coef_array
        coef_func = ngs.CoefficientFunction(grid_func)
        return coef_func.Compile()

    def vertex_inclusions_2layers_coefficient(self):
        """High coefficient inclusions around the coarse nodes with 2 layers of elements."""
        # setup piecewise constant coefficient function
        constant_fes = ngs.L2(self.two_mesh.fine_mesh, order=0)
        grid_func = ngs.GridFunction(constant_fes)

        # define background and contrast values
        background = 1.0
        contrast = 1e8

        # Get all free coarse node coordinates
        free_coarse_nodes = list(self.fes.free_component_tree_dofs.keys())

        # Loop over coarse nodes and set high contrast to 2 layers of elements around them
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

    def edge_inclusions_coefficient(self):
        """High coefficient inclusions centered on the coarse edges."""
        # setup piecewise constant coefficient function
        constant_fes = ngs.L2(self.two_mesh.fine_mesh, order=0)
        grid_func = ngs.GridFunction(constant_fes)

        # define background and contrast values
        background = 1.0
        contrast = 1e8

        # get num vertices on subdomain edges (without coarse nodes)
        num_edge_vertices = 2**self.two_mesh.refinement_levels - 2

        # set the indices of the edge vertices around which elements will be set to high contrast
        edge_vertex_inclusion_idxs = np.array(
            [i for i in range(2, num_edge_vertices, 2)]
        )

        # asssemble grid function
        coef_array = np.full(self.two_mesh.fine_mesh.ne, background)
        for coarse_edge in self.fes.free_coarse_edges:
            # get already sorted fine vertices on free coarse edge
            fine_vertices = self.two_mesh.coarse_edges_map[coarse_edge]["fine_vertices"]
            for i in edge_vertex_inclusion_idxs:
                vertex = fine_vertices[i]
                mesh_el = self.two_mesh.fine_mesh[vertex]
                for el in mesh_el.elements:
                    coef_array[el.nr] = contrast

        grid_func.vec.FV().NumPy()[:] = coef_array
        coef_func = ngs.CoefficientFunction(grid_func)
        return coef_func.Compile()

    def edge_slab_inclusions_coefficient(self):
        """High coefficient inclusions centered on the coarse edges with slabs."""
        # setup piecewise constant coefficient function
        constant_fes = ngs.L2(self.two_mesh.fine_mesh, order=0)
        grid_func = ngs.GridFunction(constant_fes)

        # define background and contrast values
        background = 1.0
        contrast = 1e8

        # construct the coefficient array
        coef_array = np.full(self.two_mesh.fine_mesh.ne, background)
        for coarse_edge in self.fes.free_coarse_edges:
            slab_elements = self.two_mesh.edge_slabs[coarse_edge.nr]
            coef_array[slab_elements] = contrast

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


class DiffusionProblemExample:
    # mesh parameters
    lx = 1.0
    ly = 1.0
    coarse_mesh_size = 1 / 16
    refinement_levels = 4
    layers = 2

    # source and coefficient functions
    source_func = SourceFunc.CONSTANT
    coef_func = CoefFunc.EDGE_SLAB_INCLUSIONS

    # preconditioner and coarse space
    preconditioner = TwoLevelSchwarzPreconditioner
    coarse_space = RGDSWCoarseSpace

    # use GPU for solving
    use_gpu = False

    # save coarse bases
    save_coarse_bases = False

    # save CG convergence information
    get_cg_info = True

    # save source, coefficient and solution as VTK files
    save_functions_toggle = False

    @classmethod
    def example_construction(cls):
        # create diffusion problem
        cls.diffusion_problem = DiffusionProblem(
            HomogeneousDirichlet(ProblemType.DIFFUSION),
            lx=cls.lx,
            ly=cls.ly,
            coarse_mesh_size=cls.coarse_mesh_size,
            refinement_levels=cls.refinement_levels,
            layers=cls.layers,
            source_func=cls.source_func,
            coef_func=cls.coef_func,
        )
        print(cls.diffusion_problem.boundary_conditions)

    @classmethod
    def example_solve(cls):
        # solve problem
        cls.diffusion_problem.solve(
            preconditioner=cls.preconditioner,
            coarse_space=cls.coarse_space,
            rtol=1e-8,
            use_gpu=cls.use_gpu,
            save_cg_info=cls.get_cg_info,
            save_coarse_bases=cls.save_coarse_bases,
        )

    @classmethod
    def save_functions(cls):
        """Save the source and coefficient functions to vtk."""
        if cls.save_functions_toggle:
            cls.diffusion_problem.save_functions()

    @classmethod
    def visualize_convergence(cls):
        if cls.get_cg_info:
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
            axs[0, 0].plot(cls.diffusion_problem.cg_alpha, label=r"$\alpha$")
            axs[0, 0].plot(cls.diffusion_problem.cg_beta, label=r"$\beta$")
            axs[0, 0].set_xlabel("Iteration")
            axs[0, 0].set_ylabel("Coefficient Value")
            axs[0, 0].legend()

            # plot residuals and preconditioned residuals
            axs[0, 1].plot(
                cls.diffusion_problem.cg_residuals, label=r"$||r_m||_2 / ||r_0||_2$"
            )
            if cls.diffusion_problem.cg_precond_residuals is not None:
                axs[0, 1].plot(
                    cls.diffusion_problem.cg_precond_residuals,
                    label=r"$||z_m||_2 / ||z_0||_2$",
                )
            axs[0, 1].set_xlabel("Iteration")
            axs[0, 1].set_ylabel("Relative residuals")
            axs[0, 1].set_yscale("log")
            axs[0, 1].legend()

            plt.suptitle(
                f"CG convergence ("
                + r"$\mathcal{C}$"
                + f" = {cls.coef_func.name}, "
                + r"$f$"
                + f" = {cls.source_func.name}, "
                + r"$M^{-1}$"
                + f" = {cls.diffusion_problem.precond_name})"
            )

            # Plot each spectrum at a different y
            axs_bottom.plot(
                np.real(cls.diffusion_problem.approximate_eigs),
                np.full_like(cls.diffusion_problem.approximate_eigs, 0),
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

            cond = np.max(cls.diffusion_problem.approximate_eigs) / np.min(
                cls.diffusion_problem.approximate_eigs
            )
            ax2.set_ylim(axs_bottom.get_ylim())
            ax2.set_yticks([0], [format_cond(cond)])

            plt.tight_layout()
            plt.show()

    @classmethod
    def get_save_dir(cls):
        """Get the directory where the problem files are saved."""
        return cls.diffusion_problem.two_mesh.save_dir


def full_example():
    DiffusionProblemExample.example_construction()
    DiffusionProblemExample.example_solve()
    DiffusionProblemExample.save_functions()
    DiffusionProblemExample.visualize_convergence()


def profile_solve():
    import cProfile
    import pstats

    from lib.utils import visualize_profile

    DiffusionProblemExample.example_construction()
    fp = DiffusionProblemExample.get_save_dir() / "diffusion_problem_solve.prof"
    cProfile.run("DiffusionProblemExample.example_solve()", str(fp))
    p = pstats.Stats(str(fp))
    p.sort_stats("cumulative").print_stats(10)
    visualize_profile(fp)


if __name__ == "__main__":
    full_example()  # Uncomment this line to run a full diffusion problem example
    # profile_solve()  # Uncomment this line to profile problem solving
