from enum import Enum
from typing import Type

import ngsolve as ngs
from boundary_conditions import (
    BoundaryCondition,
    BoundaryConditions,
    BoundaryType,
    HomogeneousDirichlet,
)
from coarse_space import GDSWCoarseSpace, Q1CoarseSpace, RGDSWCoarseSpace
from mesh import BoundaryName, TwoLevelMesh
from preconditioners import OneLevelSchwarzPreconditioner, TwoLevelSchwarzPreconditioner
from problem import Problem
from problem_type import ProblemType


class SourceFunc(Enum):
    """Enumeration for source functions used in the diffusion problem."""

    CONSTANT = "constant_source"
    PARABOLIC = "parabolic_source"


class CoefFunc(Enum):
    """Enumeration for coefficient functions used in the diffusion problem."""

    SINUSOIDAL = "sinusoidal_coefficient"
    INCLUSIONS = "inclusions_coefficient"
    CONSTANT = "constant_coefficient"
    HETMANIUK_LEHOUCQ = "hetmaniuk_lehoucq_coefficient"
    HEINLEIN = "heinlein_coefficient"


class DiffusionProblem(Problem):
    """Class representing a diffusion problem on a two-level mesh."""

    def __init__(
        self,
        boundary_conditions,
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
        c = ngs.CoefficientFunction(0.0)
        h = ngs.specialcf.mesh_size
        contrast = 1e8
        all_coarse_vertices = set(self.two_mesh.coarse_mesh.vertices)
        for subdomain, _ in self.two_mesh.subdomains.items():
            mesh_e = self.two_mesh.fine_mesh[subdomain]
            vertices = set(mesh_e.vertices)
            all_coarse_vertices -= vertices
            if len(all_coarse_vertices) == 0:
                return c
            for v in mesh_e.vertices:
                p = self.two_mesh.fine_mesh[v].point
                d = ngs.sqrt((ngs.x - p[0]) ** 2 + (ngs.y - p[1]) ** 2)
                c += ngs.IfPos(h - d, contrast, 1.0)
        return c

    def save_functions(self):
        """Save the source and coefficient functions to vtk."""
        self.save_ngs_functions(
            funcs=[self.source_func, self.coef_func, self.u],
            names=[self.source_func_name, self.coef_func_name, "solution"],
            category="diffusion",
        )


if __name__ == "__main__":
    # Example usage
    source_func = SourceFunc.PARABOLIC
    coef_func = CoefFunc.INCLUSIONS
    diffusion_problem = DiffusionProblem(
        HomogeneousDirichlet(ProblemType.DIFFUSION),
        coarse_mesh_size=0.15,
        refinement_levels=2,
        layers=2,
        source_func=source_func,
        coef_func=coef_func,
    )
    print(diffusion_problem.boundary_conditions)

    # solve problem
    get_cg_info = True
    diffusion_problem.solve(
        preconditioner=TwoLevelSchwarzPreconditioner,
        coarse_space=Q1CoarseSpace,
        rtol=1e-8,
        save_cg_info=get_cg_info,
        save_coarse_bases=False,
    )

    # Save the functions to vtk files
    # diffusion_problem.save_functions()

    # plot cg coefficients
    import matplotlib.pyplot as plt
    import numpy as np

    from lib.utils import set_mpl_cycler, set_mpl_style

    set_mpl_style()
    set_mpl_cycler(colors=True, lines=True)
    if get_cg_info:
        figure, ax = plt.subplots(1, 2, figsize=(10, 4), squeeze=True)
        ax[0].plot(diffusion_problem.cg_alpha, label=r"$\alpha$")
        ax[0].plot(diffusion_problem.cg_beta, label=r"$\beta$")
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Coefficient Value")
        ax[0].legend()
        if np.any(diffusion_problem.cg_alpha < 0):
            print(
                "Warning: Negative alpha values detected. This may indicate an issue with the preconditioner."
            )
        ax[1].plot(diffusion_problem.cg_residuals / diffusion_problem.cg_residuals[0])
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel(r"$\frac{||r_m||_2}{||r_0||_2}$")
        ax[1].set_yscale("log")

        plt.suptitle(
            f"CG convergence ("
            + r"$\mathcal{C}$"
            + f" = {coef_func.name}, "
            + r"$f$"
            + f" = {source_func.name}, "
            + r"$M^{-1}$"
            + f" = {diffusion_problem.precond_name})"
        )
        plt.tight_layout()
        plt.show()
