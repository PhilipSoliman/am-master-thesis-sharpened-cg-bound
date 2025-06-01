from datetime import datetime
from enum import Enum

import matplotlib.pyplot as plt
import ngsolve as ngs
import numpy as np
from boundary_conditions import BoundaryCondition, BoundaryConditions, BoundaryType, HomogeneousDirichlet
from fespace import FESpace
from mesh import BoundaryName, TwoLevelMesh


class Problem:
    def __init__(self, two_mesh: TwoLevelMesh, bcs: BoundaryConditions):
        """
        Initialize the problem with the given coefficient function, source function, and boundary conditions.

        Args:
            coefficient_function (callable): Function representing the coefficient in the PDE.
            source_function (callable): Function representing the source term in the PDE.
            boundary_conditions (dict): Dictionary containing boundary conditions for the problem.
        """
        self.two_mesh = two_mesh
        self.bcs = bcs
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
        if not self._linear_form_set:
            raise ValueError(
                "Linear form has not been set. Call set_linear_form() first."
            )
        if not self._bilinear_form_set:
            raise ValueError(
                "Bilinear form has not been set. Call set_bilinear_form() first."
            )

        if self._linear_form is None or self._bilinear_form is None:
            raise ValueError(
                "Linear or bilinear form has not been initialized. Call construct_fespace() first."
            )
        # solution grid function
        u = ngs.GridFunction(self.fes.fespace)

        # set boundary conditions on the grid function
        self.bcs.set_boundary_conditions_on_gfunc(u, self.fes.fespace, self.two_mesh.fine_mesh)

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

    def save_ngs_functions(self, funcs: list[ngs.GridFunction], names: list[str], category: str):
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
            # subdivision=2,
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

    # # direct solve
    # problem.direct_ngs_solve(u, b, A)

    # # save the solution
    # problem.save_ngs_solution(u, "poisson")
