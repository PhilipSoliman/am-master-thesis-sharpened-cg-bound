from enum import Enum
from typing import Optional, Type

import ngsolve as ngs
import scipy.sparse as sp
from coarse_space import CoarseSpace
from mesh import BoundaryName, TwoLevelMesh
from preconditioners import (
    OneLevelSchwarzPreconditioner,
    Preconditioner,
    TwoLevelSchwarzPreconditioner,
)
from problem import BoundaryType, Problem
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg as sp_cg


class SourceFunc(Enum):
    """Enumeration for source functions used in the diffusion problem."""

    CONSTANT = "constant_source"
    PARABOLIC = "parabolic_source"


class CoeffFunc(Enum):
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
        lx=1.0,
        ly=1.0,
        coarse_mesh_size=0.15,
        refinement_levels=2,
        layers=2,
        coeff_func=CoeffFunc.INCLUSIONS,
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

            # save mesh for reuse
            two_mesh.save()

        # initialize the Problem with the TwoLevelMesh
        super().__init__(two_mesh)

        # set boundary conditions
        self.set_boundary_condition(BoundaryName.LEFT, BoundaryType.DIRICHLET, 0.0)
        self.set_boundary_condition(BoundaryName.RIGHT, BoundaryType.DIRICHLET, 0.0)
        self.set_boundary_condition(BoundaryName.BOTTOM, BoundaryType.DIRICHLET, 0.0)
        self.set_boundary_condition(BoundaryName.TOP, BoundaryType.DIRICHLET, 0.0)

        # construct finite element space
        self.construct_fespace(order=1, discontinuous=False)

        # get trial and test functions
        u_h, v_h = self.get_trial_and_test_functions()

        # construct bilinear and linear forms
        self.coeff_func_name = coeff_func.value
        self.coeff_func = getattr(self, self.coeff_func_name)()
        self.set_bilinear_form(self.coeff_func * ngs.grad(u_h) * ngs.grad(v_h) * ngs.dx)

        self.source_func_name = source_func.value
        self.source_func = getattr(self, self.source_func_name)()
        self.set_linear_form(self.source_func * v_h * ngs.dx)

    def solve(
        self,
        preconditioner: Optional[Type[Preconditioner]] = None,
        coarse_space: Optional[CoarseSpace] = None,
        rtol: float = 1e-8,
    ):
        # assemble the system
        self.u, b, A = self.assemble()

        # export to numpy arrays
        u_arr = self.u.vec.FV().NumPy()
        b_arr = b.vec.FV().NumPy()

        # convert A to operator
        tmp1 = b.vec.CreateVector()
        tmp2 = b.vec.CreateVector()
        # TODO: need to take out boundary dofs
        def matvec(v):
            tmp1.FV().NumPy()[:] = v
            tmp2.data = A.mat * tmp1
            return tmp2.FV().NumPy()

        A_op = LinearOperator((A.mat.height, A.mat.width), matvec)

        # get preconditioner
        M_op = None
        precond = None
        if preconditioner is not None:
            raise NotImplementedError(
                "The preconditioned CG solver is not implemented yet. "
            )
            if isinstance(preconditioner, type):
                if preconditioner is OneLevelSchwarzPreconditioner:
                    precond = OneLevelSchwarzPreconditioner(self.fes, A_sp)
                elif preconditioner is TwoLevelSchwarzPreconditioner:
                    if coarse_space is None:
                        raise ValueError(
                            "Coarse space must be provided for TwoLevelSchwarzPreconditioner."
                        )
                    precond = TwoLevelSchwarzPreconditioner(
                        self.fes, A_sp, coarse_space
                    )
                else:
                    raise ValueError(
                        f"Unknown preconditioner type: {preconditioner.__name__}"
                    )
                M_op = LinearOperator(A_sp.shape, lambda x: precond.apply(x))

        # solve system using (P)CG
        u_arr[:], info = sp_cg(A_op, b_arr, x0=u_arr, M=M_op, rtol=rtol)
        if info != 0:
            raise RuntimeError(f"Conjugate gradient solver did not converge. Number of iterations: {info}")

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
        for subdomain, subdomain_data in self.two_mesh.subdomains.items():
            mesh_e = self.two_mesh.fine_mesh[subdomain]
            vertices = set(mesh_e.vertices)
            all_coarse_vertices -= vertices
            if len(all_coarse_vertices) == 0:
                print("No more coarse vertices left to process.")
                return c
            for v in mesh_e.vertices:
                p = self.two_mesh.fine_mesh[v].point
                d = ngs.sqrt((ngs.x - p[0]) ** 2 + (ngs.y - p[1]) ** 2)
                c += ngs.IfPos(h - d, contrast, 1.0)
        return c

    def save_functions(self):
        """Save the source and coefficient functions to vtk."""
        self.save_ngs_functions(
            funcs=[self.source_func, self.coeff_func, self.u],
            names=[self.source_func_name, self.coeff_func_name, "solution"],
            category="diffusion",
        )


if __name__ == "__main__":
    # Example usage
    diffusion_problem = DiffusionProblem(source_func=SourceFunc.PARABOLIC, coeff_func=CoeffFunc.CONSTANT)
    diffusion_problem.solve(
        preconditioner=None,  # Replace with actual preconditioner if needed
        coarse_space=None,  # Replace with actual coarse space if needed
        rtol=1e-8,
    )
    # Save the functions to vtk files
    diffusion_problem.save_functions()
