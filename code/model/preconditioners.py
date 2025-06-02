from typing import Type

import numpy as np
import scipy.sparse as sp
from coarse_space import CoarseSpace
from fespace import FESpace
from mesh import TwoLevelMesh
from scipy.sparse.linalg import splu


class Preconditioner(object):
    def apply(self, x: np.ndarray):
        raise NotImplementedError("This method should be implemented by subclasses.")


class OneLevelSchwarzPreconditioner(Preconditioner):
    def __init__(self, A: sp.csr_matrix, fespace: FESpace):
        free_dofs = np.array(fespace.fespace.FreeDofs())

        self.local_operators = []
        for data in fespace.domain_dofs.values():
            # get dofs on subdomain
            subdomain_dofs = np.zeros(fespace.fespace.ndof)
            subdomain_dofs[
                data["interior"] + data["edges"] + data["coarse_nodes"] + data["layer"]
            ] = True

            # take only free dofs on subdomain
            local_free_dofs = np.logical_and(free_dofs, subdomain_dofs)[free_dofs]

            # create local operator for the subdomain
            A_i = splu(A[local_free_dofs, :][:, local_free_dofs].tocsc())

            # store local free dofs and local operator
            self.local_operators.append((local_free_dofs, A_i))

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        for local_free_dofs, A_i in self.local_operators:
            out[local_free_dofs] += A_i.solve(x[local_free_dofs])
        return out


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
        self.A_0_lu = splu(self.coarse_op @ A @ self.coarse_op.transpose())

    def apply(self, x: np.ndarray):
        # First level.
        x = super().apply(x)

        # Second level.
        x_0 = self.coarse_op @ x
        y_0 = self.A_0_lu.solve(x_0)
        x += self.coarse_op.transpose() @ y_0

        return x
