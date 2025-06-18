from typing import Type

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, SuperLU, factorized
from tqdm import tqdm

from lib.fespace import FESpace
from lib.meshes import TwoLevelMesh
from lib.preconditioners import CoarseSpace


class Preconditioner(object):
    def apply(self, x: np.ndarray):
        raise NotImplementedError("This method should be implemented by subclasses.")


class OneLevelSchwarzPreconditioner(Preconditioner):
    def __init__(self, A: sp.csr_matrix, fespace: FESpace):
        print("Initializing 1-level Schwarz preconditioner")
        self.shape = A.shape
        self.free_dofs = np.array(fespace.fespace.FreeDofs()).astype(bool)
        self.fespace = fespace
        self.local_solvers = self._get_local_solvers(A)
        self.name = "1-level Schwarz preconditioner"

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        for local_free_dofs, A_i_solver in self.local_solvers:
            out[local_free_dofs] += A_i_solver(x[local_free_dofs])
        return out

    def as_linear_operator(self) -> LinearOperator:
        """Return the preconditioner as a linear operator."""
        return LinearOperator(self.shape, lambda x: self.apply(x))

    def _get_local_solvers(self, A: sp.csr_matrix) -> list[tuple[np.ndarray, SuperLU]]:
        """Get local solvers for each subdomain."""
        local_solvers = []
        free_idx = np.flatnonzero(
            self.free_dofs
        )  # Maps free dof global idx -> restricted idx
        for subdomain_dofs in tqdm(
            self.fespace.domain_dofs.values(),
            desc="Getting local solvers",
            total=len(self.fespace.domain_dofs),
        ):
            # Efficiently get local free dofs as the intersection of subdomain and free dofs
            subdomain_dofs = np.array(subdomain_dofs).astype(int)
            local_free_global = subdomain_dofs[self.free_dofs[subdomain_dofs]]

            # Map global free dofs to restricted matrix indices
            local_free_dofs = np.searchsorted(free_idx, local_free_global)

            # local operator for subdomain
            A_i_solver = factorized(A[local_free_dofs, :][:, local_free_dofs].tocsc())

            # store local free dofs and local operator
            local_solvers.append((local_free_dofs, A_i_solver))
        return local_solvers

    def __str__(self):
        return self.name


class TwoLevelSchwarzPreconditioner(OneLevelSchwarzPreconditioner):
    def __init__(
        self,
        A: sp.csr_matrix,
        fespace: FESpace,
        two_mesh: TwoLevelMesh,
        coarse_space: Type[CoarseSpace],
    ):
        super().__init__(A, fespace)
        self.coarse_space = coarse_space(A, fespace, two_mesh)
        self.name = f"2-level Schwarz preconditioner with {self.coarse_space}"
        coarse_op = self.coarse_space.assemble_coarse_operator(A)
        print("\tobtaining coarse solver")
        self.coarse_solver = factorized(coarse_op.tocsc())
        print("\tdone obtaining coarse solver")

    def apply(self, x: np.ndarray):
        # first level
        x_first_level = super().apply(x)

        # second level
        x_0 = self.coarse_space.restriction_operator.transpose() @ x
        y_0 = self.coarse_solver(x_0)
        x_second_level = self.coarse_space.restriction_operator @ y_0

        return x_first_level + x_second_level

    def get_restriction_operator_bases(self) -> dict[str, ngs.GridFunction]:
        return self.coarse_space.get_restriction_operator_bases()
