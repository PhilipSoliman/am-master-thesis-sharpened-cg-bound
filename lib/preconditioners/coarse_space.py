import warnings

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import spsolve

from lib.fespace import FESpace
from lib.meshes import TwoLevelMesh
from lib.problem_type import ProblemType

warnings.simplefilter("ignore", SparseEfficiencyWarning)


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
        self.name = "Coarse space (base)"  # to be overridden by subclasses

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
                vals = np.zeros(self.fespace.total_dofs)
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

    def _print_init_string(self):
        """
        Initialize string representation of the coarse space.
        """
        print(
            f"{self.name} initialized:"
            f"\n\tfree dofs: {self.fespace.num_free_dofs}"
            f"{self._meta_info()}"
        )

    def _meta_info(self) -> str:
        """
        Return a string with meta information about the coarse space.
        This method can be overridden by subclasses to provide more specific information.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses to provide meta information."
        )


class Q1CoarseSpace(CoarseSpace):
    def __init__(self, A: sp.csr_matrix, fespace: FESpace, two_mesh: TwoLevelMesh):
        super().__init__(A, fespace, two_mesh)
        self.name = "Q1 coarse space"
        self.restriction_operator = self.assemble_restriction_operator()
        self.coarse_dimension = self.restriction_operator.shape[1]  # type: ignore
        self._print_init_string()

    def assemble_restriction_operator(self) -> sp.csc_matrix:
        prolongation_operator = self.fespace.get_prolongation_operator()
        return prolongation_operator[self.fespace.free_dofs_mask, :]

    def _meta_info(self) -> str:
        return f"\n\tcoarse dimension: {self.coarse_dimension}"


class GDSWCoarseSpace(CoarseSpace):
    def __init__(self, A: sp.csr_matrix, fespace: FESpace, two_mesh: TwoLevelMesh):
        super().__init__(A, fespace, two_mesh)
        self.name = "GDSW coarse space"

        # null space basis and dimension
        self.null_space_dim = self._get_null_space_basis()

        # all connected components
        self.num_coarse_node_components = len(fespace.free_coarse_node_dofs)
        self.num_edge_components = len(fespace.free_edge_component_dofs)
        self.num_face_components = len(fespace.free_face_component_dofs)
        self.interface_components = self._get_connected_components()

        # interface component dimension
        self.interface_dimension = len(self.interface_components) * self.null_space_dim

        # print initialization information
        self._print_init_string()

    def assemble_restriction_operator(self) -> sp.csc_matrix:
        # get the interface operator
        interface_operator = self._assemble_interface_operator()

        # create the restriction operator
        restriction_operator = sp.csc_matrix(
            (self.fespace.num_free_dofs, self.interface_dimension), dtype=float
        )

        # interior operator
        A_II = self.A[self.fespace.interior_dofs_mask, :][
            :, ~self.fespace.interface_dofs_mask
        ]

        # interior <- interface operator
        A_IGamma = self.A[self.fespace.interior_dofs_mask, :][
            :, self.fespace.interface_dofs_mask
        ]

        # discrete harmonic extension
        interior_op = -spsolve(A_II.tocsc(), (A_IGamma @ interface_operator).tocsc())

        # fill the coarse operator #TODO: find more efficient way to construct the restriction operator
        restriction_operator[~self.fespace.interface_dofs_mask, :] = interior_op.tocsc()
        restriction_operator[self.fespace.interface_dofs_mask, :] = interface_operator
        return restriction_operator

    def _assemble_interface_operator(self):
        """
        Assemble the interface operator for the GDSW coarse space.
        Note: for problems with multiple dimensions, coordinate dofs corresponding to one node are spaced by
        ndofs (so if ndof = 8, then dofs for node 0 are 0, 8, 16, ...).
        """
        interface_operator = sp.csc_matrix(
            (self.fespace.total_dofs, self.interface_dimension),
            dtype=float,
        )
        interface_index = 0
        for interface_component in self.interface_components:
            self._restrict_null_space_to_interface_component(
                interface_operator, interface_component, interface_index
            )

            # update the interface index for the next interface component with the null space dimension
            interface_index += self.null_space_dim

        # restric the interface operator to the free and interface dofs
        return interface_operator[self.fespace.free_dofs_mask, :][
            self.fespace.interface_dofs_mask, :
        ]

    def _restrict_null_space_to_interface_component(
        self,
        interface_operator: sp.csc_matrix,
        interface_component: list[int],
        interface_index: int,
        partition_of_unity: int | np.ndarray = 1,
    ):
        """
        Restrict the null space to the interface operator for the given interface component and for all problem coordinates.
        """
        # indices of all dofs
        ndofs = self.fespace.total_dofs
        ndofs_prev = 0
        for coord, ndofs in enumerate(self.fespace.ndofs_per_unknown):
            # NOTE: NGSolve stores coordinate dofs corresponding to one node spaced by ndofs / dimension
            idxs = np.arange(ndofs_prev, ndofs_prev + ndofs)

            # get dofs for current coordinate coord
            coord_idxs_mask = np.logical_and(
                ndofs_prev < idxs[interface_component],
                idxs[interface_component] < ndofs_prev + ndofs,
            )

            # get indices of dofs for the current coordinate
            component_coord_idxs = idxs[interface_component][coord_idxs_mask]

            # get the adjusted component coordinate mask
            component_coord_mask = np.zeros(ndofs, dtype=bool)
            component_coord_mask[component_coord_idxs] = True
            component_coord_mask = component_coord_mask

            # fill the interface operator with the null space basis for the current coordinate
            # TODO: perform this construction with LIL sparse matrix for better performance
            interface_operator[
                component_coord_mask,
                interface_index : interface_index + self.null_space_dim,
            ] = self.null_space_basis[coord, :]

            # increment the previous dofs index
            ndofs_prev += ndofs

        # apply partition of unity (rowwise) and return
        interface_operator[interface_component, :] *= partition_of_unity

    def _get_null_space_basis(self):
        self.null_space_basis = np.array([])
        null_space_dim = 0
        if self.ptype == ProblemType.DIFFUSION:
            self.null_space_basis = np.ones((1, 1))
            null_space_dim = 1
        elif self.ptype == ProblemType.NAVIER_STOKES:
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
            face_component_mask = np.zeros(self.fespace.total_dofs).astype(bool)
            face_component_mask[face_component_dofs] = True
            interface_components.append(face_component_mask)

    def _get_edge_components(self, interface_components):
        for edge_component_dofs in self.fespace.free_edge_component_dofs:
            edge_component_mask = np.zeros(self.fespace.total_dofs).astype(bool)
            edge_component_mask[edge_component_dofs] = True
            interface_components.append(edge_component_mask)

    def _get_coarse_components(self, interface_components):
        for coarse_dofs in self.fespace.free_coarse_node_dofs:
            coarse_node_dofs_mask = np.zeros(self.fespace.total_dofs).astype(bool)
            coarse_node_dofs_mask[coarse_dofs] = True
            interface_components.append(coarse_node_dofs_mask)

    def _meta_info(self) -> str:
        return (
            f"\n\tnull space dim: {self.null_space_dim}"
            f"\n\tinterface dim: {self.interface_dimension}"
            f"\n\t\tcoarse node components: {self.num_coarse_node_components}"
            f"\n\t\tedge components: {self.num_edge_components}"
            f"\n\t\tface components: {self.num_face_components}"
        )


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

        # print initialization information
        self._print_init_string()

    def _assemble_interface_operator(self):
        ndofs = self.fespace.total_dofs
        interface_operator = sp.csc_matrix(
            (ndofs, self.interface_dimension),
            dtype=float,
        )
        interface_index = 0
        for component_dofs in self.component_tree_dofs.values():
            node_dofs = component_dofs["node"]
            edge_components = component_dofs["edges"]
            self._restrict_null_space_to_interface_component(
                interface_operator, node_dofs, interface_index
            )

            for edge, edge_dofs in edge_components.items():
                multiplicity = self.edge_component_multiplicities[edge]
                self._restrict_null_space_to_interface_component(
                    interface_operator, edge_dofs, interface_index, 1 / multiplicity
                )

            # update the interface index for the next interface component with the null space dimension
            interface_index += self.null_space_dim

        # restric the interface operator to the free and interface dofs
        return interface_operator[self.fespace.free_dofs_mask, :][
            self.fespace.interface_dofs_mask, :
        ]

    def _meta_info(self) -> str:
        return (
            f"\n\tnull space dim: {self.null_space_dim}"
            f"\n\tinterface dim: {self.interface_dimension}"
        )


class AMSCoarseSpace(GDSWCoarseSpace):
    def __init__(self, A: sp.csr_matrix, fespace: FESpace, two_mesh: TwoLevelMesh):
        CoarseSpace.__init__(self, A, fespace, two_mesh)
        self.name = "AMS coarse space"
        self.coarse_dofs_mask, self.edge_dofs_mask, self.face_dofs_mask = (
            self._get_interface_component_masks()
        )
        self.interface_dimension = np.sum(self.coarse_dofs_mask)
        self.num_coarse_dofs = self.interface_dimension
        self.num_edge_dofs = np.sum(self.edge_dofs_mask)
        self.num_face_dofs = np.sum(self.face_dofs_mask)

        self.interior_dofs_mask = ~self.fespace.interface_dofs_mask
        self._print_init_string()

    def assemble_restriction_operator(self) -> sp.csc_matrix:
        restriction_operator = sp.csc_matrix(
            (self.fespace.num_free_dofs, self.interface_dimension), dtype=float
        )

        # Phi_V
        vertex_restriction = sp.eye(self.interface_dimension, dtype=float)

        # Phi_E
        A_EE = self.A[self.edge_dofs_mask, :][:, self.edge_dofs_mask]
        A_EI = self.A[self.edge_dofs_mask, :][:, self.fespace.interior_dofs_mask]
        A_EE += sp.diags(A_EI.sum(axis=1).A1, offsets=0, format="csc")
        if np.any(self.face_dofs_mask):
            A_EF = self.A[self.edge_dofs_mask, :][:, self.face_dofs_mask]
            A_EE += sp.diags(A_EF.sum(axis=1).A1, offsets=0, format="csc")
        A_EV = self.A[self.edge_dofs_mask, :][:, self.coarse_dofs_mask]
        edge_restriction = -spsolve(A_EE.tocsc(), A_EV.tocsc())

        # Phi_F
        face_restriction = sp.csc_matrix(
            (self.num_face_dofs, self.interface_dimension), dtype=float
        )
        if np.any(self.face_dofs_mask):
            A_FF = self.A[self.face_dofs_mask, :][:, self.face_dofs_mask]
            A_FI = self.A[self.face_dofs_mask, :][:, self.fespace.interior_dofs_mask]
            A_FF += sp.diags(A_FI.sum(axis=1).A1, offsets=0, format="csc")
            A_FV = self.A[self.face_dofs_mask, :][:, self.coarse_dofs_mask]
            A_FE = self.A[self.face_dofs_mask, :][:, self.edge_dofs_mask]
            face_restriction = -spsolve(
                A_FF.tocsc(),
                (A_FV @ vertex_restriction + A_FE @ edge_restriction).tocsc(),
            )

        # Phi_I
        A_II = self.A[self.interior_dofs_mask, :][:, self.interior_dofs_mask]
        A_IV = self.A[self.interior_dofs_mask, :][:, self.coarse_dofs_mask]
        A_IE = self.A[self.interior_dofs_mask, :][:, self.edge_dofs_mask]
        A_IF = self.A[self.interior_dofs_mask, :][:, self.face_dofs_mask]
        interface_restriction = (
            A_IV @ vertex_restriction
            + A_IE @ edge_restriction
            + A_IF @ face_restriction
        )
        interior_restriction = -spsolve(
            A_II.tocsc(),
            interface_restriction.tocsc(),
        )

        # collect all restrictions at their respective dofs
        restriction_operator[self.coarse_dofs_mask, :] = vertex_restriction
        restriction_operator[self.edge_dofs_mask, :] = edge_restriction
        if np.any(self.face_dofs_mask):
            restriction_operator[self.face_dofs_mask, :] = face_restriction
        restriction_operator[self.interior_dofs_mask, :] = interior_restriction

        return restriction_operator

    def _get_interface_component_masks(self):
        # get all groups of interface components
        coarse_components = []
        super()._get_coarse_components(coarse_components)
        coarse_dofs_mask = np.zeros(self.fespace.total_dofs).astype(bool)
        for coarse_component in coarse_components:
            coarse_dofs_mask[coarse_component] = True

        edge_components = []
        super()._get_edge_components(edge_components)
        edge_dofs_mask = np.zeros(self.fespace.total_dofs).astype(bool)
        for edge_component in edge_components:
            edge_dofs_mask[edge_component] = True

        face_components = []
        super()._get_face_components(face_components)
        face_dofs_mask = np.zeros(self.fespace.total_dofs).astype(bool)
        for face_component in face_components:
            face_dofs_mask[face_component] = True

        return (
            coarse_dofs_mask[self.fespace.free_dofs_mask],
            edge_dofs_mask[self.fespace.free_dofs_mask],
            face_dofs_mask[self.fespace.free_dofs_mask],
        )

    def _meta_info(self) -> str:
        return (
            f"\n\tinterface dimension: {self.interface_dimension}"
            f"\n\tcoarse dofs: {self.num_coarse_dofs}"
            f"\n\tedge dofs: {self.num_edge_dofs}"
            f"\n\tface dofs: {self.num_face_dofs}"
        )
