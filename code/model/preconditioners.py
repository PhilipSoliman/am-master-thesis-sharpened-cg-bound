from pprint import pprint
from typing import Type

import numpy as np
import scipy.sparse as sp
from coarse_space import CoarseSpace
from fespace import FESpace
from matplotlib import pyplot as plt
from mesh import TwoLevelMesh
from scipy.sparse.linalg import SuperLU, splu


class Preconditioner(object):
    def apply(self, x: np.ndarray):
        raise NotImplementedError("This method should be implemented by subclasses.")


class OneLevelSchwarzPreconditioner(Preconditioner):
    def __init__(self, A: sp.csr_matrix, fespace: FESpace):
        self.free_dofs = np.array(fespace.fespace.FreeDofs()).astype(bool)
        self.fespace = fespace
        self.local_operators = self._get_local_operators(A)

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        for local_free_dofs, A_i in self.local_operators:
            out[local_free_dofs] += A_i.solve(x[local_free_dofs])
        return out

    def _get_local_operators(
        self, A: sp.csr_matrix
    ) -> list[tuple[np.ndarray, SuperLU]]:
        """Get local operators for each subdomain."""
        local_operators = []
        for subdomain_dofs in self.fespace.domain_dofs.values():
            # get dofs on subdomain
            subdomain_mask = np.zeros(self.fespace.fespace.ndof).astype(bool)
            subdomain_mask[self._get_all_subdomain_dofs(subdomain_dofs)] = True

            # take only free dofs on subdomain
            local_free_dofs = subdomain_mask[self.free_dofs]

            # local operator for subdomain
            A_i = splu(A[local_free_dofs, :][:, local_free_dofs].tocsc())

            # store local free dofs and local operator
            local_operators.append((local_free_dofs, A_i))
        return local_operators

    def _get_all_subdomain_dofs(self, subdomain_dofs):
        """Get all subdomain dofs."""
        all_subdomain_dofs = set()
        for component_type, component in subdomain_dofs.items():
            if (
                component_type == "interior"
                or component_type == "coarse_nodes"
                or component_type == "layer"
            ):
                all_subdomain_dofs.update(component)
            elif component_type == "edges":
                for edge_nr, edge_dofs in component.items():
                    all_subdomain_dofs.update(edge_dofs["vertices"])
                    all_subdomain_dofs.update(edge_dofs["edges"])
            elif component_type == "face":
                raise NotImplementedError("Face dofs are not implemented.")
        return list(all_subdomain_dofs)


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
        A_0 = (self.coarse_op.transpose() @ (A @ self.coarse_op))
        self.A_0_lu = splu(A_0.tocsc())

    def apply(self, x: np.ndarray):
        # First level.
        x = super().apply(x)

        # Second level.
        x_0 = self.coarse_op.transpose() @ x
        y_0 = self.A_0_lu.solve(x_0)
        x += self.coarse_op @ y_0

        return x

    def _get_coarse_operator_gfuncs(self):
        coarse_op_gfuncs = {"edge": [], "coarse_node": []}

        # get coarse operator grid functions for coarse nodes
        for comp in range(self.coarse_space.num_coarse_node_components):
            vals = np.zeros(self.fespace.fespace.ndof)
            vals[self.free_dofs] = self.coarse_op[:, comp].toarray().flatten()
            gfunc = self.coarse_space.fespace.get_gridfunc(vals)
            coarse_op_gfuncs["coarse_node"].append(gfunc)

        # ... and for edges
        for comp in range(
            self.coarse_space.num_coarse_node_components,
            self.coarse_space.num_edge_components,
        ):
            vals = np.zeros(self.fespace.fespace.ndof)
            vals[self.free_dofs] = self.coarse_op[:, comp].toarray().flatten()
            gfunc = self.coarse_space.fespace.get_gridfunc(vals)
            coarse_op_gfuncs["edge"].append(gfunc)

        return coarse_op_gfuncs
