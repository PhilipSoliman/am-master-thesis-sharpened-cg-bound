from copy import copy
from datetime import datetime
from typing import Optional, Type

import matplotlib.pyplot as plt
import ngsolve as ngs
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from lib.boundary_conditions import (
    BoundaryCondition,
    BoundaryConditions,
    BoundaryType,
    HomogeneousDirichlet,
)
from lib.fespace import FESpace
from lib.meshes import BoundaryName, TwoLevelMesh
from lib.preconditioners import (
    CoarseSpace,
    OneLevelSchwarzPreconditioner,
    Preconditioner,
    TwoLevelSchwarzPreconditioner,
)
from lib.problem_type import ProblemType
from lib.solvers import CustomCG


class Problem:
    def __init__(
        self,
        two_mesh: TwoLevelMesh,
        bcs: list[BoundaryConditions],
        ptype: ProblemType = ProblemType.CUSTOM,
    ):
        """
        Initialize the problem with the given coefficient function, source function, and boundary conditions.

        Args:
            coefficient_function (callable): Function representing the coefficient in the PDE.
            source_function (callable): Function representing the source term in the PDE.
            boundary_conditions (dict): Dictionary containing boundary conditions for the problem.
        """
        self.two_mesh = two_mesh
        self.boundary_conditions = bcs
        self.ptype = ptype
        self._linear_form = None
        self._linear_form_set = False
        self._bilinear_form = None
        self._bilinear_form_set = False

    def construct_fespace(
        self,
        fespaces: Optional[list] = None,
        orders: Optional[list] = None,
        dimensions: Optional[list] = None,
    ):
        """
        Construct the finite element space for the problem.

        Returns:
            FESpace: The finite element space for the problem.
        """
        if self.ptype == ProblemType.CUSTOM:
            if fespaces is None or orders is None or dimensions is None:
                raise ValueError(
                    "For custom problem type, fespaces, orders, and dimensions must be provided."
                )
            self.ptype.fespaces = fespaces
            self.ptype.orders = orders
            self.ptype.dimensions = dimensions
        self.fes = FESpace(
            self.two_mesh,
            self.boundary_conditions,
            ptype=self.ptype,
        )
        print(f"Constructed FESpace for {str(self.ptype)}")
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
        u = self.fes.u  # type: ignore
        v = self.fes.v  # type: ignore
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

    def assemble(self, gfuncs: Optional[list[ngs.GridFunction]] = None):
        """
        Assemble the linear and bilinear forms.
        """
        # check if linear and bilinear forms are instantiated on the finite element space
        if self._linear_form is None or self._bilinear_form is None:
            raise ValueError(
                "Linear or bilinear form has not been initialized. Call construct_fespace() first."
            )

        # check if problem type is set
        if self.ptype == ProblemType.CUSTOM:
            if not self._linear_form_set:
                raise ValueError(
                    "Linear form has not been set. Call set_linear_form() first."
                )
            if not self._bilinear_form_set:
                raise ValueError(
                    "Bilinear form has not been set. Call set_bilinear_form() first."
                )
        elif self.ptype == ProblemType.DIFFUSION:
            # check if gfuncs is provided for custom coefficient and source functions
            if gfuncs is None or len(gfuncs) != 2:
                raise ValueError(
                    "For diffusion problem type, gfuncs must be provided with two grid functions."
                    "One for coefficient function and one for source function."
                )

            # get trial and test functions
            u_h, v_h = self.get_trial_and_test_functions()

            # construct bilinear and linear forms
            self.set_bilinear_form(gfuncs[0] * ngs.grad(u_h) * ngs.grad(v_h) * ngs.dx)
            self.set_linear_form(gfuncs[1] * v_h * ngs.dx)
        elif self.ptype == ProblemType.NAVIER_STOKES:
            raise NotImplementedError(
                "Navier-Stokes problem type is not implemented yet."
            )
        else:
            raise ValueError(
                f"Problem type {self.ptype} is not supported. Use ProblemType.CUSTOM."
            )

        # solution grid function
        u = ngs.GridFunction(self.fes.fespace)

        # set boundary conditions on the grid function
        for bcs in self.boundary_conditions:
            bcs.set_boundary_conditions_on_gfunc(
                u, self.fes.fespace, self.two_mesh.fine_mesh
            )

        # assemble rhs and stiffness matrix
        b = self._linear_form.Assemble()
        A = self._bilinear_form.Assemble()

        return A, u, b

    def get_homogenized_system(
        self, A: ngs.Matrix, u: ngs.GridFunction, b: ngs.GridFunction
    ):
        res = b.vec.CreateVector()
        res.data = b.vec - A.mat * u.vec

        # free dofs
        free_dofs = self.fes.fespace.FreeDofs()

        # export to numpy arrays & sparse matrix
        u_arr = copy(u.vec.FV().NumPy()[free_dofs])
        res_arr = res.FV().NumPy()[free_dofs]
        rows, cols, vals = A.mat.COO()
        A_sp = sp.csr_matrix((vals, (rows, cols)), shape=A.mat.shape)
        A_sp_f = A_sp[free_dofs, :][:, free_dofs]

        return A_sp_f, u_arr, res_arr

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
        save_coarse_bases: bool = False,
    ):
        # assemble the system
        A, self.u, b = self.assemble()

        # homogenization of the boundary conditions
        A_sp_f, u_arr, res_arr = self.get_homogenized_system(A, self.u, b)

        # get preconditioner
        M_op = None
        precond = None
        coarse_space_bases = {}
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
                    if save_coarse_bases:
                        coarse_space_bases = precond.get_restriction_operator_bases()
                else:
                    raise ValueError(
                        f"Unknown preconditioner type: {preconditioner.__name__}"
                    )
                M_op = precond.as_linear_operator()
        self.precond_name = precond.name if precond is not None else "None"

        # solve system using (P)CG
        custom_cg = CustomCG(
            A_sp_f,
            res_arr,
            u_arr,
            tol=rtol,
        )
        print(f"Solving system:" f"\n\tpreconditioner: {self.precond_name}")
        u_arr[:], success = custom_cg.sparse_solve(M=M_op, save_residuals=save_cg_info)
        if not success:
            print(
                f"Conjugate gradient solver did not converge. Number of iterations: {custom_cg.niters}"
            )
        else:
            self.u.vec.FV().NumPy()[self.fes.fespace.FreeDofs()] = u_arr

        # save cg coefficients if requested
        if save_cg_info:
            self.cg_alpha, self.cg_beta = custom_cg.alpha, custom_cg.beta
            self.cg_residuals = custom_cg.get_relative_residuals()
            self.cg_precond_residuals = None
            if precond is not None:
                self.cg_precond_residuals = (
                    custom_cg.get_relative_preconditioned_residuals()
                )
            self.approximate_eigs = custom_cg.get_approximate_eigenvalues()

        # save coarse operator grid functions if available
        if save_coarse_bases and coarse_space is not None:
            gfuncs = []
            names = []
            for basis, basis_gfunc in coarse_space_bases.items():
                names.append(basis)
                gfuncs.append(basis_gfunc)
            if gfuncs != []:
                self.save_ngs_functions(gfuncs, names, "coarse_bases")
            else:
                print("No coarse space bases to save.")

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

    # define problem type
    ptype = ProblemType.DIFFUSION

    # define boundary conditions
    bcs = HomogeneousDirichlet(ptype)
    bcs.set_boundary_condition(
        BoundaryCondition(
            name=BoundaryName.LEFT,
            btype=BoundaryType.DIRICHLET,
            values={0: 32 * ngs.y * (ly - ngs.y)},
        )
    )
    print(bcs)

    # construct finite element space
    problem = Problem(two_mesh, [bcs])

    # construct finite element space
    problem.construct_fespace([ngs.H1], [1], [1])

    # get trial and test functions
    u_h, v_h = problem.get_trial_and_test_functions()

    # construct bilinear and linear forms
    problem.set_bilinear_form(ngs.grad(u_h) * ngs.grad(v_h) * ngs.dx)
    problem.set_linear_form(
        32 * (ngs.y * (ly - ngs.y) + ngs.x * (lx - ngs.x)) * v_h * ngs.dx
    )

    # assemble the forms
    A, u, b = problem.assemble()

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
    # problem.save_ngs_functions([u], ["solution"], "test_solution_direct")

    # cg solve
    problem.solve()
    # problem.save_ngs_functions([problem.u], ["solution"], "test_solution_cg")
