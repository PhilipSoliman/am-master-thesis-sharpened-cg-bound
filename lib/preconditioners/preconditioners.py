from typing import Type

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, SuperLU, splu

from lib.fespace import FESpace
from lib.meshes import TwoLevelMesh
from lib.preconditioners import CoarseSpace


class Preconditioner(object):
    def apply(self, x: np.ndarray):
        raise NotImplementedError("This method should be implemented by subclasses.")


class OneLevelSchwarzPreconditioner(Preconditioner):
    def __init__(self, A: sp.csr_matrix, fespace: FESpace):
        self.shape = A.shape
        self.free_dofs = np.array(fespace.fespace.FreeDofs()).astype(bool)
        self.fespace = fespace
        self.local_operators = self._get_local_operators(A)
        self.name = "1-level Schwarz preconditioner"

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        for local_free_dofs, A_i in self.local_operators:
            out[local_free_dofs] += A_i.solve(x[local_free_dofs])
        return out
    
    def as_linear_operator(self) -> LinearOperator:
        """Return the preconditioner as a linear operator."""
        return LinearOperator(self.shape, lambda x: self.apply(x))

    def _get_local_operators(
        self, A: sp.csr_matrix
    ) -> list[tuple[np.ndarray, SuperLU]]:
        """Get local operators for each subdomain."""
        local_operators = []
        for subdomain_dofs in self.fespace.domain_dofs.values():
            # get dofs on subdomain
            subdomain_mask = np.zeros(self.fespace.fespace.ndof).astype(bool)
            subdomain_mask[subdomain_dofs] = True

            # take only free dofs on subdomain
            local_free_dofs = subdomain_mask[self.free_dofs]

            # local operator for subdomain
            A_i = splu(A[local_free_dofs, :][:, local_free_dofs].tocsc())

            # store local free dofs and local operator
            local_operators.append((local_free_dofs, A_i))
        return local_operators

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
        self.coarse_op = self.coarse_space.assemble_coarse_operator(A)

    def apply(self, x: np.ndarray):
        # first level
        x_first_level = super().apply(x)

        # second level
        x_0 = self.coarse_space.restriction_operator.transpose() @ x
        y_0 = splu(self.coarse_op.tocsc()).solve(x_0)
        x_second_level = self.coarse_space.restriction_operator @ y_0

        return x_first_level + x_second_level

    def get_restriction_operator_bases(self) -> dict[str, ngs.GridFunction]:
        return self.coarse_space.get_restriction_operator_bases()
