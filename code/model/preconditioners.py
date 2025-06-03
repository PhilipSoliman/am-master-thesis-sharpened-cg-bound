from typing import Type

import numpy as np
import scipy.sparse as sp
from coarse_space import CoarseSpace
from fespace import FESpace
from matplotlib import pyplot as plt
from mesh import TwoLevelMesh
from scipy.sparse.linalg import splu


class Preconditioner(object):
    def apply(self, x: np.ndarray):
        raise NotImplementedError("This method should be implemented by subclasses.")


class OneLevelSchwarzPreconditioner(Preconditioner):
    def __init__(self, A: sp.csr_matrix, fespace: FESpace):
        self.free_dofs = np.array(fespace.fespace.FreeDofs()).astype(bool)
        self.fespace = fespace
        self.local_operators = []

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        for local_free_dofs, A_i in self.local_operators:
            out[local_free_dofs] += A_i.solve(x[local_free_dofs])
        return out

    def _get_local_operators(self, A: sp.csr_matrix):
        """Get local operators for each subdomain."""
        for data in self.fespace.domain_dofs.values():
            # get dofs on subdomain
            subdomain_dofs = np.zeros(self.fespace.fespace.ndof)
            subdomain_dofs[self._get_all_subdomain_dofs()] = True

            # take only free dofs on subdomain
            local_free_dofs = np.logical_and(self.free_dofs, subdomain_dofs)[
                self.free_dofs
            ]

            # create local operator for the subdomain
            A_i = splu(A[local_free_dofs, :][:, local_free_dofs].tocsc())

            # store local free dofs and local operator
            self.local_operators.append((local_free_dofs, A_i))

    def _get_all_subdomain_dofs(self):
        """Get all subdomain dofs."""
        all_subdomain_dofs = set()
        for data in self.fespace.domain_dofs.values():
            all_subdomain_dofs.update(
                data["interior"] + data["coarse_nodes"] + data["layer"]
            )
            for edge_dofs in data["edges"].values():
                all_subdomain_dofs.update(edge_dofs) 
        return all_subdomain_dofs


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
        self.coarse_op = self.coarse_space.assemble_coarse_operator()
        # plt.spy(self.coarse_op.toarray(), markersize=1, aspect="equal")
        # plt.title("Sparsity pattern of the coarse operator Phi")
        # plt.xlabel("Columns")
        # plt.ylabel("Rows")
        # plt.show()
        rank = np.linalg.matrix_rank(self.coarse_op.toarray())
        print("Rank of coarse operator:", rank)
        A_0 = (self.coarse_op.transpose() @ (A @ self.coarse_op)).tocsc()
        # plt.spy(A_0.toarray(), markersize=1, aspect="equal")
        # plt.title("Sparsity pattern of the coarse operator A_0")
        # plt.xlabel("Columns")
        # plt.ylabel("Rows")
        # plt.show()
        print("Shape of A_0:", A_0.shape)
        print("Rank of A_0:", np.linalg.matrix_rank(A_0.toarray()))

        self.A_0_lu = splu(A_0)

    def apply(self, x: np.ndarray):
        # First level.
        x = super().apply(x)

        # Second level.
        x_0 = self.coarse_op @ x
        y_0 = self.A_0_lu.solve(x_0)
        x += self.coarse_op.transpose() @ y_0

        return x
