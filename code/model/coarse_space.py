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

        # collect all interface dofs
        self.interface_dofs = set()
        self.interface_dofs.update(fespace.coarse_node_dofs)
        self.interface_dofs.update(fespace.edge_dofs)

        # create a mask for free interface dofs
        self.interface_dofs_mask = np.zeros(self.fespace.fespace.ndof, dtype=bool)
        self.interface_dofs_mask[list(self.interface_dofs)] = True
        self.interface_dofs_mask = np.logical_and(
            self.free_dofs_mask, self.interface_dofs_mask
        )[self.free_dofs_mask]

        # null space basis and dimension
        self.null_space_dim = self._get_null_space_basis()

        # all connected components
        self.interface_components = self._get_connected_components()

        # interface component dimension
        self.interface_dim = len(self.interface_components) * self.null_space_dim

        # print important meta information
        print(
            f"Coarse space initialized:"
            f"\n\tproblem type: {self.ptype.value}"
            f"\n\tfinite element space: {self.fespace.dimension}D "
            f"\n\tfree dofs: {self.num_free_dofs}"
            f"\n\tnull space dim: {self.null_space_dim}"
            f"\n\tinterface components: {len(self.interface_components)}"
            f"\n\tinterface dim: {self.interface_dim}"
        )

    def assemble_coarse_operator(self) -> sp.csc_matrix:
        raise NotImplementedError("This method should be implemented by subclasses.")

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

    def _get_connected_components(self):
        interface_components = []

        # face components
        if self.fespace.num_face_dofs > 0:
            self._get_face_components(interface_components)

        # edge components
        if self.fespace.num_edge_dofs > 0:
            self._get_edge_components(interface_components)

        # coarse node components
        if self.fespace.num_coarse_node_dofs > 0:
            self._get_coarse_components(interface_components)

        return interface_components

    def _get_face_components(self, interface_components):
        raise NotImplementedError(
            "Face components are not implemented for this coarse space."
        )

    def _get_edge_components(self, interface_components):
        num_subdomain_edges = len(self.two_mesh.coarse_mesh.edges)
        edge_nrs = set(range(num_subdomain_edges))
        num_edge_components = 0
        for data in self.fespace.domain_dofs.values():
            for edge_nr, coarse_edge_dofs in data["edges"].items():
                # check if edge_nr is not already processed
                if int(edge_nr) not in edge_nrs:
                    continue

                # create a mask for the coarse edge dofs
                edge_component_mask = np.zeros(self.fespace.fespace.ndof)
                edge_component_mask[coarse_edge_dofs] = True

                # only add components if it contains free dofs
                if np.any(edge_component_mask[self.free_dofs_mask]):
                    interface_components.append(edge_component_mask)
                    num_edge_components += 1

                # remove edge_nr from the list of edge nrs to process
                edge_nrs.remove(int(edge_nr))



        print(f"found {num_edge_components} edge components")

    def _get_coarse_components(self, interface_components):
        all_coarse_node_dofs = self.fespace.coarse_node_dofs.copy()
        num_coarse_node_components = 0
        for data in self.fespace.domain_dofs.values():
            for coarse_node_dof in data["coarse_nodes"]:
                # check if coarse node dof is not already processed
                if coarse_node_dof not in all_coarse_node_dofs:
                    continue

                # create a mask for the coarse node dof
                coarse_node_dofs_mask = np.zeros(self.fespace.fespace.ndof)
                coarse_node_dofs_mask[coarse_node_dof] = True

                # only add components if it contains free dofs
                if np.any(coarse_node_dofs_mask[self.free_dofs_mask]):
                    interface_components.append(coarse_node_dofs_mask)
                    num_coarse_node_components += 1

                # remove coarse node dof from the list of coarse node dofs to process
                all_coarse_node_dofs.remove(coarse_node_dof)

        print(f"found {num_coarse_node_components} coarse node components")

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
        for interface_component in self.interface_components:
            # get dofs for the current interface component
            component_mask = np.logical_and(
                self.free_dofs_mask, interface_component
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

        # restric the interface operator to the free and interface dofs
        self.interface_op = self.interface_op[self.free_dofs_mask, :][
            self.interface_dofs_mask, :
        ]


class RGDSWCoarseSpace(CoarseSpace):
    def assemble_coarse_operator(self) -> sp.csc_matrix:
        # Implement the RGDSW coarse space application logic here
        raise NotImplementedError("RGDSW coarse space application not implemented.")
