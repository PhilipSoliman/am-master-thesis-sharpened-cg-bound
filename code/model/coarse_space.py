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
        self.num_free_dofs = np.sum(self.fespace.fespace.FreeDofs())
        self.free_dofs_mask = np.array(self.fespace.fespace.FreeDofs()).astype(bool)

        # get the null space basis and dimension
        self.null_space_dim = self._get_null_space_basis()

        # get all connected components and meta info
        self.all_interface_components_dofs = set()
        self.interface_components_dofs = []
        self.interface_dim, self.interface_dofs_mask = (
            self._get_connected_components()
        )

    def assemble_coarse_operator(self) -> sp.csc_matrix:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _get_connected_components(self):
        for data in self.fespace.domain_dofs.values():
            # face components
            if (subdomain_face_dofs := data.get("faces")) is not None:
                self._get_face_components(subdomain_face_dofs)

            # edge components
            if (subdomain_coarse_nodes_dofs := data.get("edges")) is not None:
                self._get_edge_components(subdomain_coarse_nodes_dofs)

            # coarse node dofs
            if (subdomain_coarse_nodes_dofs := data.get("coarse_nodes")) is not None:
                self._get_coarse_components(subdomain_coarse_nodes_dofs)

        # collect meta info about interface components
        num_interface_components = len(self.interface_components_dofs)

        # derive interface dimension
        interface_dim = self.null_space_dim * num_interface_components

        # mask for interface dofs
        interface_dofs_mask = np.zeros(self.fespace.fespace.ndof, dtype=bool)
        interface_dofs_mask[list(self.all_interface_components_dofs)] = True
        interface_dofs_mask = np.logical_and(self.free_dofs_mask, interface_dofs_mask)[
            self.free_dofs_mask
        ]
        return interface_dim, interface_dofs_mask

    def _get_null_space_basis(self):
        self.null_space_basis = np.array([])
        null_space_dim = 0
        if self.ptype == ProblemType.DIFFUSION:
            self.null_space_basis = np.ones((1, 1))
            null_space_dim = 1
        elif self.ptype == ProblemType.ADVECTION:
            raise NotImplementedError(
                "Advection problem type not implemented for coarse space."
            )
        else:
            raise ValueError(f"Unsupported problem type: {self.ptype}")
        return null_space_dim

    def _get_face_components(self, subdomain_face_dofs):
        raise NotImplementedError(
            "Face components are not implemented for this coarse space."
        )

    def _get_edge_components(self, subdomain_edges_dofs):
        for nr, coarse_edge_dofs in subdomain_edges_dofs.items():
            edge_component = np.zeros(self.fespace.fespace.ndof)
            edge_component[coarse_edge_dofs] = True
            remaining_edge_dofs = list(
                set(coarse_edge_dofs) - self.all_interface_components_dofs
            )
            edge_component[remaining_edge_dofs] = True
            self.all_interface_components_dofs.update(remaining_edge_dofs)

            edge_component_mask = np.logical_and(self.free_dofs_mask, edge_component)
            if np.any(edge_component_mask):
                print(f"coarse edge {nr} component dofs:", np.where(edge_component_mask)[0])
                self.interface_components_dofs.append(edge_component_mask)

    def _get_coarse_components(self, subdomain_coarse_nodes_dofs):
        coarse_node_dofs = np.zeros(self.fespace.fespace.ndof)
        remaining_coarse_node_dofs = list(
            set(subdomain_coarse_nodes_dofs) - self.all_interface_components_dofs
        )
        coarse_node_dofs[remaining_coarse_node_dofs] = True
        self.all_interface_components_dofs.update(remaining_coarse_node_dofs)

        coarse_node_dofs_mask = np.logical_and(self.free_dofs_mask, coarse_node_dofs)
        if np.any(coarse_node_dofs_mask):
            print("coarse node component dofs:", np.where(coarse_node_dofs_mask)[0])
            self.interface_components_dofs.append(coarse_node_dofs_mask)


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
        coarse_op = sp.csc_matrix((self.num_free_dofs, self.interface_dim), dtype=float)

        # interior operator
        A_II = self.A[~self.interface_dofs_mask, :][:, ~self.interface_dofs_mask]

        # interior <- interface operator
        A_IGamma = self.A[~self.interface_dofs_mask, :][:, self.interface_dofs_mask]

        # discrete harmonic extension
        interior_op = -spsolve(A_II, (A_IGamma @ self.interface_op).tocsc())

        # fill the coarse operator
        coarse_op[~self.interface_dofs_mask, :] = interior_op
        coarse_op[self.interface_dofs_mask, :] = self.interface_op
        return coarse_op

    def _assemble_interface_operator(self):
        """
        Assemble the interface operator for the GDSW coarse space.
        Note: for problems with multiple dimensions, coordinate dofs corresponding to one node are spaced by
        ndofs (so if ndof = 8, then dofs for node 0 are 0, 8, 16, ...).
        """
        ndofs = self.fespace.fespace.ndof
        self.interface_op = sp.csc_matrix(
            (ndofs, self.interface_dim),
            dtype=float,
        )
        idxs = np.arange(ndofs)
        interface_index = 0
        for interface_component_dofs in self.interface_components_dofs:
            # get dofs for the current interface component
            component_mask = np.logical_and(
                self.free_dofs_mask, interface_component_dofs
            )
            for coord in range(self.fespace.dimension):
                # get dofs for current coordinate coord
                coord_idxs_mask = np.logical_and(
                    coord < idxs[component_mask],
                    idxs[component_mask] < (coord + 1) * ndofs,
                )
                # get indices of dofs for the current coordinate
                component_coord_idxs = idxs[component_mask][coord_idxs_mask]

                # get the adjusted component coordinate mask
                component_coord_mask = np.zeros(ndofs, dtype=bool)
                component_coord_mask[component_coord_idxs] = True
                component_coord_mask = component_coord_mask

                # fill the interface operator with the null space basis for the current coordinate
                self.interface_op[
                    component_coord_mask,
                    interface_index : interface_index + self.null_space_dim,
                ] = self.null_space_basis[coord, :]
            # update the interface index for the next interface component with the null space dimension
            interface_index += self.null_space_dim

        self.interface_op = self.interface_op[self.free_dofs_mask, :][
            self.interface_dofs_mask, :
        ]


class RGDSWCoarseSpace(CoarseSpace):
    def assemble_coarse_operator(self) -> sp.csc_matrix:
        # Implement the RGDSW coarse space application logic here
        raise NotImplementedError("RGDSW coarse space application not implemented.")
