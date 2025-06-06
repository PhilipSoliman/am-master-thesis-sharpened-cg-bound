import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
from fespace import FESpace
from mesh import TwoLevelMesh
from problem_type import ProblemType
from scipy.sparse.linalg import spsolve


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

        self.name = "base coarse space"  # to be overridden by subclasses

        # print important meta information
        print(
            f"Coarse space initialized:"
            f"\n\tproblem type: {self.ptype.value}"
            f"\n\tfinite element space: {self.fespace.dimension}D "
            f"\n\tfree dofs: {self.fespace.num_free_dofs}"
        )

    def assemble_coarse_operator(self, A: sp.csr_matrix) -> sp.csc_matrix:
        if not hasattr(self, "restriction_operator"):
            self.restriction_operator = self.assemble_restriction_operator()
        return self.restriction_operator.transpose() @ (A @ self.restriction_operator)

    def assemble_restriction_operator(self) -> sp.csc_matrix:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_restriction_operator_bases(self) -> dict[str, ngs.GridFunction]:
        if (
            restriction_operator := getattr(self, "restriction_operator", None)
        ) is not None:
            num_bases = restriction_operator.shape[1]
            bases = {}
            for i in range(num_bases):
                vals = np.zeros(self.fespace.fespace.ndof)
                vals[self.fespace.free_dofs_mask] = (
                    restriction_operator[:, i].toarray().flatten()
                )
                gfunc = self.fespace.get_gridfunc(vals)
                bases[f"coarse_basis_{i}"] = gfunc
            return bases
        else:
            raise ValueError("Restriction operator is not assembled yet.")

    def __str__(self):
        return self.name


class Q1CoarseSpace(CoarseSpace):
    def __init__(self, A: sp.csr_matrix, fespace: FESpace, two_mesh: TwoLevelMesh):
        super().__init__(A, fespace, two_mesh)
        self.name = "Q1 coarse space"
        self.restriction_operator = self.assemble_restriction_operator()

    def assemble_restriction_operator(self) -> sp.csc_matrix:
        return self.fespace.prolongation_operator[self.fespace.free_dofs_mask, :]


class GDSWCoarseSpace(CoarseSpace):
    def __init__(self, A: sp.csr_matrix, fespace: FESpace, two_mesh: TwoLevelMesh):
        super().__init__(A, fespace, two_mesh)
        self.name = "GDSW coarse space"

        # collect all interface dofs
        self.interface_dofs_mask = self.fespace.interface_dofs_mask

        # null space basis and dimension
        self.null_space_dim = self._get_null_space_basis()

        # all connected components
        self.num_coarse_node_components = len(fespace.free_coarse_node_dofs)
        self.num_edge_components = len(fespace.free_edge_component_dofs)
        self.num_face_components = len(fespace.free_face_component_dofs)
        self.interface_components = self._get_connected_components()

        # interface component dimension
        self.interface_dimension = len(self.interface_components) * self.null_space_dim

        # print important meta information
        print(
            f"GDSW coarse space initialized:"
            f"\n\tnull space dim: {self.null_space_dim}"
            f"\n\tinterface components: {len(self.interface_components)}"
            f"\n\t\tcoarse node components: {self.num_coarse_node_components}"
            f"\n\t\tedge components: {self.num_edge_components}"
            f"\n\t\tface components: {self.num_face_components}"
            f"\n\tinterface dim: {self.interface_dimension}"
        )

    def assemble_restriction_operator(self) -> sp.csc_matrix:
        # get the interface operator
        interface_operator = self._assemble_interface_operator()

        # create the restriction operator
        restriction_operator = sp.csc_matrix(
            (self.fespace.num_free_dofs, self.interface_dimension), dtype=float
        )

        # interior operator
        A_II = self.A[~self.fespace.interface_dofs_mask, :][
            :, ~self.fespace.interface_dofs_mask
        ]

        # interior <- interface operator
        A_IGamma = self.A[~self.fespace.interface_dofs_mask, :][
            :, self.fespace.interface_dofs_mask
        ]

        # discrete harmonic extension
        interior_op = -spsolve(A_II.tocsc(), (A_IGamma @ interface_operator).tocsc())

        # fill the coarse operator #TODO: find more efficient way to do this
        restriction_operator[~self.fespace.interface_dofs_mask, :] = interior_op.tocsc()
        restriction_operator[self.fespace.interface_dofs_mask, :] = interface_operator
        return restriction_operator

    def _assemble_interface_operator(self):
        """
        Assemble the interface operator for the GDSW coarse space.
        Note: for problems with multiple dimensions, coordinate dofs corresponding to one node are spaced by
        ndofs (so if ndof = 8, then dofs for node 0 are 0, 8, 16, ...).
        """
        # NOTE: NGSolve stores coordinate dofs corresponding to one node spaced by ndofs / dimension
        ndofs = self.fespace.fespace.ndof
        coord_diff = ndofs // self.fespace.dimension
        idxs = np.arange(ndofs)
        interface_operator = sp.csc_matrix(
            (ndofs, self.interface_dimension),
            dtype=float,
        )
        interface_index = 0
        for interface_component in self.interface_components:
            # get dofs for the current interface component
            for coord in range(self.fespace.dimension):
                # get dofs for current coordinate coord
                coord_idxs_mask = np.logical_and(
                    coord < idxs[interface_component],
                    idxs[interface_component] < (coord + 1) * coord_diff,
                )

                # get indices of dofs for the current coordinate
                component_coord_idxs = idxs[interface_component][coord_idxs_mask]

                # get the adjusted component coordinate mask
                component_coord_mask = np.zeros(ndofs, dtype=bool)
                component_coord_mask[component_coord_idxs] = True
                component_coord_mask = component_coord_mask

                # fill the interface operator with the null space basis for the current coordinate
                # TODO: perform this construction LIL sparse matrix for better performance
                interface_operator[
                    component_coord_mask,
                    interface_index : interface_index + self.null_space_dim,
                ] = self.null_space_basis[coord, :]

            # update the interface index for the next interface component with the null space dimension
            interface_index += self.null_space_dim

        # restric the interface operator to the free and interface dofs
        return interface_operator[self.fespace.free_dofs_mask, :][
            self.fespace.interface_dofs_mask, :
        ]

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
        # coarse node components
        if len(self.fespace.free_coarse_node_dofs) > 0:
            self._get_coarse_components(interface_components)

        # edge components
        if len(self.fespace.free_edge_component_dofs) > 0:
            self._get_edge_components(interface_components)

        # face components
        if len(self.fespace.free_face_component_dofs) > 0:
            self._get_face_components(interface_components)

        return interface_components

    def _get_face_components(self, interface_components):
        for face_component_dofs in self.fespace.free_face_component_dofs:
            face_component_mask = np.zeros(self.fespace.fespace.ndof).astype(bool)
            face_component_mask[face_component_dofs] = True
            interface_components.append(face_component_mask)

    def _get_edge_components(self, interface_components):
        for edge_component_dofs in self.fespace.free_edge_component_dofs:
            edge_component_mask = np.zeros(self.fespace.fespace.ndof).astype(bool)
            edge_component_mask[edge_component_dofs] = True
            interface_components.append(edge_component_mask)

    def _get_coarse_components(self, interface_components):
        for coarse_dofs in self.fespace.free_coarse_node_dofs:
            coarse_node_dofs_mask = np.zeros(self.fespace.fespace.ndof).astype(bool)
            coarse_node_dofs_mask[coarse_dofs] = True
            interface_components.append(coarse_node_dofs_mask)


class RGDSWCoarseSpace(GDSWCoarseSpace):
    def __init__(self, A: sp.csr_matrix, fespace: FESpace, two_mesh: TwoLevelMesh):
        CoarseSpace.__init__(self, A, fespace, two_mesh)
        self.name = "RGDSW coarse space"

        # null space basis and dimension
        self.null_space_dim = self._get_null_space_basis()

        # interface components
        self.component_tree_dofs = self.fespace.free_component_tree_dofs
        self.edge_component_multiplicities = self.fespace.edge_component_multiplicities

        # interface dimension
        self.interface_dimension = len(self.component_tree_dofs) * self.null_space_dim
        if self.interface_dimension == 0:
            raise ValueError("No RGDSW components with coarse node ancestors found")

        # print important meta information
        print(
            f"GDSW coarse space initialized:"
            f"\n\tnull space dim: {self.null_space_dim}"
            f"\n\tinterface components: {len(self.component_tree_dofs)}"
            f"\n\tinterface dim: {self.interface_dimension}"
        )

    def _assemble_interface_operator(self):
        ndofs = self.fespace.fespace.ndof
        coord_diff = ndofs // self.fespace.dimension
        idxs = np.arange(ndofs)
        interface_operator = sp.csc_matrix(
            (ndofs, self.interface_dimension),
            dtype=float,
        )
        interface_index = 0
        for coarse_node, component_dofs in self.component_tree_dofs.items():
            node_dofs = component_dofs["node"]
            edge_components = component_dofs["edges"]
            for coord in range(self.fespace.dimension):
                ################
                # coarse nodes #
                ################
                # get coarse dofs for current coordinate coord
                node_coord_idxs_mask = np.logical_and(
                    coord < idxs[node_dofs],
                    idxs[node_dofs] < (coord + 1) * coord_diff,
                )

                # get indices of dofs for the current coordinate
                component_coord_idxs = idxs[node_dofs][node_coord_idxs_mask]

                # get the adjusted component coordinate mask
                component_coord_mask = np.zeros(ndofs, dtype=bool)
                component_coord_mask[component_coord_idxs] = True
                component_coord_mask = component_coord_mask

                interface_operator[
                    component_coord_mask,
                    interface_index : interface_index + self.null_space_dim,
                ] = self.null_space_basis[coord, :]

                for edge, edge_dofs in edge_components.items():
                    ##############
                    # edge nodes #
                    ##############
                    # get multiplicity for the current edge
                    multiplicity = self.edge_component_multiplicities[edge]

                    # get dofs for current coordinate coord
                    coord_idxs_mask = np.logical_and(
                        coord < idxs[edge_dofs],
                        idxs[edge_dofs] < (coord + 1) * coord_diff,
                    )

                    # get indices of dofs for the current coordinate
                    component_coord_idxs = idxs[edge_dofs][coord_idxs_mask]

                    # get the adjusted component coordinate mask
                    component_coord_mask = np.zeros(ndofs, dtype=bool)
                    component_coord_mask[component_coord_idxs] = True
                    component_coord_mask = component_coord_mask

                    # fill the interface operator with the null space basis for the current coordinate
                    # TODO: perform this construction LIL sparse matrix for better performance
                    interface_operator[
                        component_coord_mask,
                        interface_index : interface_index + self.null_space_dim,
                    ] = (
                        self.null_space_basis[coord, :]
                        / multiplicity  # divide by multiplicity (a.k.k. Option 1 RGDSW, 2017)
                    )

            # update the interface index for the next interface component with the null space dimension
            interface_index += self.null_space_dim

        # restric the interface operator to the free and interface dofs
        return interface_operator[self.fespace.free_dofs_mask, :][
            self.fespace.interface_dofs_mask, :
        ]


class AMSCoarseSpace(GDSWCoarseSpace):
    def assemble_restriction_operator(self) -> sp.csc_matrix:
        # Implement the AMS coarse space application logic here
        raise NotImplementedError("AMS coarse space application not implemented.")
