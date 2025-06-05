from copy import copy
from datetime import datetime
from enum import Enum
from typing import Optional, Type

import matplotlib.pyplot as plt
import ngsolve as ngs
import scipy.sparse as sp
from boundary_conditions import (
    BoundaryCondition,
    BoundaryConditions,
    BoundaryType,
    HomogeneousDirichlet,
)
from coarse_space import CoarseSpace
from fespace import FESpace
from mesh import BoundaryName, TwoLevelMesh
from preconditioners import (
    OneLevelSchwarzPreconditioner,
    Preconditioner,
    TwoLevelSchwarzPreconditioner,
)
from problem_type import ProblemType
from scipy.sparse.linalg import LinearOperator

from lib.custom_cg import CustomCG


class Problem:
    def __init__(
        self,
        two_mesh: TwoLevelMesh,
        bcs: BoundaryConditions,
        ptype: Optional[ProblemType] = None,
    ):
        """
        Initialize the problem with the given coefficient function, source function, and boundary conditions.

        Args:
            coefficient_function (callable): Function representing the coefficient in the PDE.
            source_function (callable): Function representing the source term in the PDE.
            boundary_conditions (dict): Dictionary containing boundary conditions for the problem.
        """
        self.two_mesh = two_mesh
        self.bcs = bcs
        self.ptype = ptype
        self._linear_form = None
        self._linear_form_set = False
        self._bilinear_form = None
        self._bilinear_form_set = False

    def construct_fespace(self, order: int = 1, discontinuous: bool = False):
        """
        Construct the finite element space for the problem.

        Args:
            order (int, optional): Polynomial order of the finite elements. Defaults to 1.
            discontinuous (bool, optional): Whether to use discontinuous elements. Defaults to False.

        Returns:
            FESpace: The finite element space for the problem.
        """
        self.fes = FESpace(
            self.two_mesh,
            order=order,
            discontinuous=discontinuous,
            **self.bcs.boundary_kwargs,
        )
        self._linear_form = ngs.LinearForm(self.fes.fespace)
        self._bilinear_form = ngs.BilinearForm(self.fes.fespace, symmetric=True)

    def get_trial_and_test_functions(self):
        """
        Get the trial and test functions for the finite element space.

        Returns:
            tuple: A tuple containing the trial and test functions.
        """
        if hasattr(self, "fes") is False:
            raise ValueError(
                "Finite element space has not been constructed yet. Call construct_fespace() first."
            )
        u = self.fes.u
        v = self.fes.v
        return u, v

    @property
    def linear_form(self):
        return self._linear_form

    @linear_form.setter
    def linear_form(self, lf: ngs.LinearForm):
        if self._linear_form is None:
            raise ValueError(
                "Linear form has not been initialized. Call construct_fespace() first."
            )
        if isinstance(self._linear_form, ngs.LinearForm):
            raise ValueError("Linear form must be set using set_linear_form().")
        self._linear_form = lf

    def set_linear_form(self, lf: ngs.LinearForm):
        """
        Set the linear form for the finite element space.

        Args:
            lf (ngs.LinearForm): The linear form to set.
        """
        self._linear_form_set = True
        self._linear_form += lf

    @property
    def bilinear_form(self):
        return self._bilinear_form

    @bilinear_form.setter
    def bilinear_form(self, bf: ngs.BilinearForm):
        if self._bilinear_form is None:
            raise ValueError(
                "Bilinear form has not been initialized. Call construct_fespace() first."
            )
        if isinstance(self._bilinear_form, ngs.BilinearForm):
            raise ValueError("Bilinear form must be set using set_bilinear_form().")
        self._bilinear_form = bf

    def set_bilinear_form(self, bf: ngs.BilinearForm):
        """
        Set the bilinear form for the finite element space.

        Args:
            bf (ngs.FESpace.BilinearForm): The bilinear form to set.
        """
        self._bilinear_form_set = True
        self._bilinear_form += bf

    def assemble(self):
        """
        Assemble the linear and bilinear forms.
        """
        # check if linear and bilinear forms are instantiated on the finite element space
        if self._linear_form is None or self._bilinear_form is None:
            raise ValueError(
                "Linear or bilinear form has not been initialized. Call construct_fespace() first."
            )

        # check if problem type is set
        if self.ptype is None:
            if not self._linear_form_set:
                raise ValueError(
                    "Linear form has not been set. Call set_linear_form() first."
                )
            if not self._bilinear_form_set:
                raise ValueError(
                    "Bilinear form has not been set. Call set_bilinear_form() first."
                )
        else:
            # TODO: Auto construct (bi-)linear forms using a (mix of) problem type(s)
            pass

        # solution grid function
        u = ngs.GridFunction(self.fes.fespace)

        # set boundary conditions on the grid function
        self.bcs.set_boundary_conditions_on_gfunc(
            u, self.fes.fespace, self.two_mesh.fine_mesh
        )

        # assemble rhs and stiffness matrix
        b = self._linear_form.Assemble()
        A = self._bilinear_form.Assemble()

        return u, b, A

    def direct_ngs_solve(self, u: ngs.Vector, b: ngs.Vector, A: ngs.Matrix):
        """
        Solve the system of equations directly.

        Args:
            gfu (ngs.GridFunction): The grid function to store the solution.
            system (ngs.Matrix): The system matrix.
            load (ngs.Vector): The load vector.
        """
        # homogenization
        res = b.vec.CreateVector()
        res.data = b.vec - A.mat * u.vec

        # solve the system using sparse Cholesky factorization
        u.vec.data += (
            A.mat.Inverse(self.fes.fespace.FreeDofs(), inverse="sparsecholesky") * res
        )

    def solve(
        self,
        preconditioner: Optional[Type[Preconditioner]] = None,
        coarse_space: Optional[Type[CoarseSpace]] = None,
        rtol: float = 1e-8,
        save_cg_info: bool = False,
        save_coarse_gfuncs: bool = False,
    ):
        # assemble the system
        self.u, b, A = self.assemble()

        # homogenization of the boundary conditions
        res = b.vec.CreateVector()
        res.data = b.vec - A.mat * self.u.vec

        # free dofs
        free_dofs = self.fes.fespace.FreeDofs()

        # export to numpy arrays & sparse matrix
        u_arr = copy(self.u.vec.FV().NumPy()[free_dofs])
        res_arr = res.FV().NumPy()[free_dofs]
        rows, cols, vals = A.mat.COO()
        A_sp = sp.csr_matrix((vals, (rows, cols)), shape=A.mat.shape)
        A_sp_f = A_sp[free_dofs, :][:, free_dofs]

        # get preconditioner
        M_op = None
        precond = None
        restriction_op_gfuncs = {}
        if preconditioner is not None:
            if isinstance(preconditioner, type):
                if preconditioner is OneLevelSchwarzPreconditioner:
                    precond = OneLevelSchwarzPreconditioner(A_sp_f, self.fes)
                elif preconditioner is TwoLevelSchwarzPreconditioner:
                    if coarse_space is None:
                        raise ValueError(
                            "Coarse space must be provided for TwoLevelSchwarzPreconditioner."
                        )
                    precond = TwoLevelSchwarzPreconditioner(
                        A_sp_f, self.fes, self.two_mesh, coarse_space
                    )
                    restriction_op_gfuncs = precond.get_restriction_operator_bases()
                else:
                    raise ValueError(
                        f"Unknown preconditioner type: {preconditioner.__name__}"
                    )
                M_op = LinearOperator(A_sp_f.shape, lambda x: precond.apply(x))
        self.precond_name = precond.name if precond is not None else "None"

        # solve system using (P)CG
        custom_cg = CustomCG(
            A_sp_f,
            res_arr,
            u_arr,
            tol=rtol,
        )
        u_arr[:], info = custom_cg.sparse_solve(
            M=M_op, save_coefficients=save_cg_info, save_residuals=save_cg_info
        )
        if info != 0:
            print(
                f"Conjugate gradient solver did not converge. Number of iterations: {info}"
            )
        else:
            self.u.vec.FV().NumPy()[free_dofs] = u_arr

        # save cg coefficients if requested
        if save_cg_info:
            self.cg_alpha, self.cg_beta = custom_cg.alpha, custom_cg.beta
            self.cg_residuals = custom_cg.r_i

        # save coarse operator grid functions if available
        if save_coarse_gfuncs and coarse_space is not None:
            gfuncs = []
            names = []
            for basis, basis_gfunc in restriction_op_gfuncs.items():
                names.append(basis)
                gfuncs.append(basis_gfunc)
            self.save_ngs_functions(gfuncs, names, "restriction_operators")

    def save_ngs_functions(
        self, funcs: list[ngs.GridFunction], names: list[str], category: str
    ):
        """
        Save the grid function solution to a file.

        Args:
            sol (ngs.GridFunction): The grid function containing the solution.
            name (str): The name of the problem (used for the filename).
        """
        # Get current date and time as a string
        now_str = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
        fn = f"{category}_{now_str}"
        tlm_dir = self.two_mesh.save_dir
        vtk = ngs.VTKOutput(
            self.two_mesh.fine_mesh,
            coefs=funcs,
            names=names,
            filename=str(tlm_dir / fn),
        )
        vtk.Do()


if __name__ == "__main__":
    # load mesh
    lx, ly = 1.0, 1.0
    coarse_mesh_size = 0.15
    refinement_levels = 2
    layers = 2
    two_mesh = TwoLevelMesh.load(
        lx=lx,
        ly=ly,
        coarse_mesh_size=coarse_mesh_size,
        refinement_levels=refinement_levels,
        layers=layers,
    )

    # define boundary conditions
    bcs = HomogeneousDirichlet()
    bcs.set_boundary_condition(
        BoundaryCondition(
            name=BoundaryName.LEFT,
            btype=BoundaryType.DIRICHLET,
            value=32 * ngs.y * (ly - ngs.y),
        )
    )
    print(bcs)

    # construct finite element space
    problem = Problem(two_mesh, bcs)

    # construct finite element space
    problem.construct_fespace(order=1, discontinuous=False)

    # get trial and test functions
    u_h, v_h = problem.get_trial_and_test_functions()

    # construct bilinear and linear forms
    problem.set_bilinear_form(ngs.grad(u_h) * ngs.grad(v_h) * ngs.dx)
    problem.set_linear_form(
        32 * (ngs.y * (ly - ngs.y) + ngs.x * (lx - ngs.x)) * v_h * ngs.dx
    )

    # assemble the forms
    u, b, A = problem.assemble()

    # plot the sparsity pattern of the stiffness matrix
    import scipy.sparse as sp

    rows, cols, vals = A.mat.COO()
    A_sp = sp.csr_matrix((vals, (rows, cols)), shape=A.mat.shape)
    plt.spy(A_sp.toarray(), markersize=1, aspect="equal")
    plt.title("Sparsity pattern of the stiffness matrix")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()

    # direct solve
    problem.direct_ngs_solve(u, b, A)
    problem.save_ngs_functions([u], ["solution"], "test_solution_direct")

    # cg solve
    problem.solve()
    problem.save_ngs_functions([problem.u], ["solution"], "test_solution_cg")
