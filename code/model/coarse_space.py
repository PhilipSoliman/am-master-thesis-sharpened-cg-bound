from copy import copy

import numpy as np
import scipy.sparse as sp
from fespace import FESpace
from mesh import TwoLevelMesh
from problem_type import ProblemType
from scipy.sparse.linalg import splu, spsolve


class CoarseSpace(object):
    def __init__(
        self,
        A: sp.csr_matrix,
        fespace: FESpace,
        two_mesh: TwoLevelMesh,
        ptype: ProblemType = ProblemType.DIFFUSION,
    ):

        self.fespace = fespace
        self.A = A
        self.two_mesh = two_mesh
        self.ptype = ptype
        self._get_connected_components()
        self._get_null_space_basis()

    def assemble_coarse_operator(self) -> sp.csc_matrix:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _get_connected_components(self):
        free_dofs = np.array(self.fespace.fespace.FreeDofs())

        self.interface_components_dofs = {"edges": [], "coarse_nodes": []}
        self.all_interface_components_dofs = set()
        for data in self.fespace.domain_dofs.values():
            # edge dofs
            edge_dofs = np.zeros(self.fespace.fespace.ndof)
            remaining_edge_dofs = list(
                set(data["edges"]) - self.all_interface_components_dofs
            )
            edge_dofs[remaining_edge_dofs] = True
            self.all_interface_components_dofs.update(remaining_edge_dofs)

            # coarse node dofs
            coarse_node_dofs = np.zeros(self.fespace.fespace.ndof)
            remaining_coarse_node_dofs = list(
                set(data["coarse_nodes"]) - self.all_interface_components_dofs
            )
            coarse_node_dofs[remaining_coarse_node_dofs] = True
            self.all_interface_components_dofs.update(remaining_coarse_node_dofs)

            # take only free dofs on edge and coarse node dofs
            if len(remaining_edge_dofs) > 0:
                edge_dofs_mask = np.logical_and(free_dofs, edge_dofs)[free_dofs]
                self.interface_components_dofs["edges"].append(edge_dofs_mask)
                self.num_interface_components_dofs += len(remaining_edge_dofs)

            if len(remaining_coarse_node_dofs) > 0:
                coarse_node_dofs_mask = np.logical_and(free_dofs, coarse_node_dofs)[
                    free_dofs
                ]
                self.interface_components_dofs["coarse_nodes"].append(
                    coarse_node_dofs_mask
                )
                self.num_interface_components_dofs += len(remaining_coarse_node_dofs)

        self.num_interface_dofs = len(self.all_interface_components_dofs)

        # mask for interface dofs
        interface_dofs_mask = np.zeros(self.fespace.fespace.ndof, dtype=bool)
        interface_dofs_mask[list(self.all_interface_components_dofs)] = True
        self.interface_dofs_mask = np.logical_and(free_dofs, interface_dofs_mask)[
            free_dofs
        ]

    def _get_null_space_basis(self):
        self.null_space_basis = np.array([])
        if self.ptype == ProblemType.DIFFUSION:
            self.null_space_basis = np.array(
                [1]
            )  # constant function for diffusion problems
        elif self.ptype == ProblemType.ADVECTION:
            raise NotImplementedError(
                "Advection problem type not implemented for coarse space."
            )
        else:
            raise ValueError(f"Unsupported problem type: {self.ptype}")


class AMSCoarseSpace(CoarseSpace):
    def assemble_coarse_operator(self) -> sp.csc_matrix:
        # Implement the AMS coarse space application logic here
        raise NotImplementedError("AMS coarse space application not implemented.")


class GDSWCoarseSpace(CoarseSpace):
    def __init__(self, A: sp.csr_matrix, fespace: FESpace, two_mesh: TwoLevelMesh):
        super().__init__(A, fespace, two_mesh)
        self._assemble_interface_operator()

    def assemble_coarse_operator(self) -> sp.csc_matrix:
        # Assemble the coarse operator A0
        A_IGamma = self.A[~self.interface_dofs_mask, :][:, self.interface_dofs_mask]
        A_II = self.A[self.interface_dofs_mask, :][:, self.interface_dofs_mask]
        Phi_I = -spsolve(A_II, A_IGamma @ self.interface_op)
        return self.interface_op.T @ A_IGamma + A_II @ self.interface_op

    def _assemble_interface_operator(self):
        problem_dim, null_space_dim = self.null_space_basis.shape
        num_free_dofs = np.sum(self.fespace.fespace.FreeDofs())
        self.interface_op = sp.csc_matrix(
            (num_free_dofs, null_space_dim * self.num_interface_dofs),
            dtype=float,
        )
        interface_index = 0
        interface_components_dofs = (
            self.interface_components_dofs["coarse_nodes"]
            + self.interface_components_dofs["edges"]
        )
        for interface_component_dofs in interface_components_dofs:
            for coord in range(problem_dim):
                # assume coordinate dofs are ordered next to each other
                idxs = np.arange(coord, len(interface_component_dofs), problem_dim)

                # interface component dofs corresponding to one coordinate
                interface_component_dofs_coord = copy(interface_component_dofs)
                interface_component_dofs_coord[~idxs] = False

                self.interface_op[
                    interface_component_dofs_coord,
                    interface_index : interface_index + null_space_dim,
                ] = self.null_space_basis[coord, :]
            interface_index += null_space_dim


class RGDSWCoarseSpace(CoarseSpace):
    def assemble_coarse_operator(self) -> sp.csc_matrix:
        # Implement the RGDSW coarse space application logic here
        raise NotImplementedError("RGDSW coarse space application not implemented.")
