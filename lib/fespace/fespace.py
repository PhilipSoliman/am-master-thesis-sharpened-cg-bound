import ngsolve as ngs
import numpy as np
import scipy.sparse as sp

from lib.boundary_conditions import BoundaryConditions, HomogeneousDirichlet
from lib.meshes import TwoLevelMesh
from lib.problem_type import ProblemType


class FESpace:
    """
    FESpace
    A class for constructing and managing a finite element space on a TwoLevelMesh,
    including the classification of DOFs (degrees of freedom) into interior, edge, and coarse node categories.
    """

    def __init__(
        self,
        two_mesh: TwoLevelMesh,
        boundary_conditions: list[BoundaryConditions],
        ptype: ProblemType = ProblemType.CUSTOM,
    ):
        """
        Initialize the finite element space for the given mesh.

        Args:
            two_mesh (TwoLevelMesh): The mesh to create the finite element space for.
            order (int, optional): The polynomial order of the finite elements. Defaults to 1.
            discontinuous (bool, optional): Whether to use discontinuous elements. Defaults to False.

        Raises:
            ValueError: If the sum of classified DOFs does not match the total number of DOFs in the space.
        """
        self.two_mesh = two_mesh
        self.ptype = ptype
        self.ndofs_per_unknown = []

        # construct fespace to get dofs
        for fespace, order, dim, bcs in zip(
            ptype.fespaces, ptype.orders, ptype.dimensions, boundary_conditions
        ):
            fespace = fespace(two_mesh.fine_mesh, order=order, **bcs.boundary_kwargs)
            for _ in range(dim):
                self.ndofs_per_unknown.append(fespace.ndof // dim)
            if hasattr(self, "fespace"):
                self.fespace *= fespace
            else:
                self.fespace = fespace

        self.calculate_dofs()
        if self.fespace.ndof != (
            self.num_interior_dofs + self.num_edge_dofs + self.num_coarse_node_dofs
        ):
            raise ValueError(
                "Mismatch in number of DOFs: "
                f"{self.fespace.ndof} != {self.num_interior_dofs} + {self.num_edge_dofs} + {self.num_coarse_node_dofs}"
            )

    def calculate_dofs(self):
        """
        Calculate and classify the degrees of freedom (DOFs) in the finite element space.

        This method:
            - Collects all DOFs from the fine mesh.
            - Calls calculate_subdomain_dofs() to classify DOFs by subdomain.
            - Removes edge and coarse node DOFs from the set of interior DOFs.
            - Stores the classified DOFs and prints a summary.
        """
        # all DOFS
        self.total_dofs = self.fespace.ndof

        # free DOFs mask
        self.free_dofs_mask = np.array(self.fespace.FreeDofs()).astype(bool)
        self.num_free_dofs = np.sum(self.free_dofs_mask)

        # get all degrees of freedom (DOFs)
        self.interior_dofs = set()
        for el in self.fespace.Elements():
            dofs = self.fespace.GetDofNrs(el)
            self.interior_dofs.update(dofs)

        # calculate subdomain DOFs
        self.domain_dofs = self.calculate_subdomain_dofs()

        # remove coarse node DOFs from fine DOFs
        self.coarse_node_dofs = set(
            self.fespace.GetDofNrs(v)[0] for v in self.two_mesh.coarse_mesh.vertices
        )
        self.edge_dofs = set()
        for subdomain_data in self.domain_dofs.values():
            for coarse_edge_dofs in subdomain_data["edges"].values():
                self.edge_dofs.update(coarse_edge_dofs["vertices"])
                self.edge_dofs.update(coarse_edge_dofs["edges"])

        # remove edge DOFs from interior DOFs
        self.interior_dofs -= self.edge_dofs
        self.interior_dofs -= set(self.coarse_node_dofs)

        # component dofs
        self.free_coarse_node_dofs = self.get_free_coarse_node_dofs()
        self.free_edge_component_dofs = self.get_free_edge_component_dofs()
        self.free_face_component_dofs = self.get_free_face_component_dofs()
        self.free_component_tree_dofs, self.edge_component_multiplicities = (
            self.get_free_component_tree_dofs()
        )

        # create a mask for free interface dofs
        interface_dofs = []
        for component_dofs in self.free_coarse_node_dofs:
            interface_dofs.extend(component_dofs)
        for component_dofs in self.free_edge_component_dofs:
            interface_dofs.extend(component_dofs)
        for component_dofs in self.free_face_component_dofs:
            interface_dofs.extend(component_dofs)
        self.interface_dofs_mask = np.zeros(self.fespace.ndof, dtype=bool)
        self.interface_dofs_mask[list(interface_dofs)] = True
        self.interface_dofs_mask = np.logical_and(
            self.free_dofs_mask, self.interface_dofs_mask
        )[self.free_dofs_mask]

        # create mask for interior dofs
        self.interior_dofs_mask = ~self.interface_dofs_mask

        # meta info
        self.num_interior_dofs = len(self.interior_dofs)
        self.num_edge_dofs = len(self.edge_dofs)
        self.num_coarse_node_dofs = len(self.coarse_node_dofs)

    def calculate_subdomain_dofs(self):
        """
        Calculate DOFs for each subdomain (coarse element) and classify them as interior, edge, coarse node, or layer DOFs.

        Returns:
            dict: Mapping from coarse elements to their classified DOFs.
        """
        domain_dofs = {}
        for subdomain, subdomain_data in self.two_mesh.subdomains.items():
            coarse_node_dofs = set()
            for v in subdomain.vertices:
                dofs = self.fespace.GetDofNrs(v)
                coarse_node_dofs.update(dofs)

            interior_dofs = set()
            for el in subdomain_data["interior"]:
                dofs = self.fespace.GetDofNrs(el)
                interior_dofs.update(dofs)
            interior_dofs -= coarse_node_dofs

            subdomain_edge_dofs = {}
            all_subdomain_edge_dofs = set()
            for coarse_edge, fine_edges in subdomain_data["edges"].items():
                subdomain_edge_dofs[coarse_edge] = {}
                coarse_edge_fine_vertex_dofs = set()
                coarse_edge_fine_edge_dofs = set()
                for e in fine_edges:
                    v1, v2 = self.two_mesh.fine_mesh[e].vertices

                    # get vertex DOFs
                    coarse_edge_fine_vertex_dofs.update(self.fespace.GetDofNrs(v1))
                    coarse_edge_fine_vertex_dofs.update(self.fespace.GetDofNrs(v2))

                    # get edge subdomain_edge_fine_edge_dofs
                    coarse_edge_fine_edge_dofs.update(self.fespace.GetDofNrs(e))
                coarse_edge_fine_vertex_dofs -= coarse_node_dofs
                interior_dofs -= coarse_edge_fine_vertex_dofs
                interior_dofs -= coarse_edge_fine_edge_dofs
                subdomain_edge_dofs[coarse_edge]["vertices"] = list(
                    coarse_edge_fine_vertex_dofs
                )
                subdomain_edge_dofs[coarse_edge]["edges"] = list(
                    coarse_edge_fine_edge_dofs
                )
                all_subdomain_edge_dofs.update(coarse_edge_fine_vertex_dofs)
                all_subdomain_edge_dofs.update(coarse_edge_fine_edge_dofs)

            layer_dofs = set()
            for layer_idx in range(1, self.two_mesh.layers + 1):
                layer_elements = subdomain_data[f"layer_{layer_idx}"]
                for el in layer_elements:
                    dofs = self.fespace.GetDofNrs(el)
                    layer_dofs.update(dofs)
            layer_dofs -= all_subdomain_edge_dofs
            layer_dofs -= coarse_node_dofs
            layer_dofs -= interior_dofs

            domain_dofs[subdomain] = {
                "interior": list(interior_dofs),
                "coarse_nodes": list(coarse_node_dofs),
                "edges": subdomain_edge_dofs,
                "layer": list(layer_dofs),
            }
        return domain_dofs

    def get_free_coarse_node_dofs(self):
        """
        Get the degrees of freedom (DOFs) associated with coarse nodes.

        Returns:
            list: A list of DOFs corresponding to coarse nodes.
        """
        coarse_node_dofs = []
        all_dofs = np.arange(self.fespace.ndof)
        free_dofs = all_dofs[self.fespace.FreeDofs()]
        for coarse_node in self.two_mesh.connected_components["coarse_nodes"]:
            dofs = set(self.fespace.GetDofNrs(coarse_node))
            dofs = dofs.intersection(free_dofs)
            if len(dofs) > 0:
                coarse_node_dofs.append(list(dofs))

        # filter coarse node dofs to only include free dofs
        return coarse_node_dofs

    def get_free_edge_component_dofs(self):
        edge_component_dofs = []
        all_dofs = np.arange(self.fespace.ndof)
        free_dofs = all_dofs[self.fespace.FreeDofs()]
        for edge_component in self.two_mesh.connected_components["edges"]:
            dofs = set()
            for c in edge_component:
                dofs.update(self.fespace.GetDofNrs(c))

            # remove boundary dofs
            dofs = [d for d in dofs if d in free_dofs]
            if len(dofs) > 0:
                edge_component_dofs.append(list(dofs))
        return edge_component_dofs

    def get_free_face_component_dofs(self):
        face_component_dofs = []
        all_dofs = np.arange(self.fespace.ndof)
        free_dofs = all_dofs[self.fespace.FreeDofs()]
        for face_component in self.two_mesh.connected_components["faces"]:
            dofs = set()
            for c in face_component:
                dofs.update(self.fespace.GetDofNrs(c))

            # remove boundary dofs
            dofs = [d for d in dofs if d in free_dofs]
            if len(dofs) > 0:
                face_component_dofs.append(list(dofs))
        return face_component_dofs

    def get_free_component_tree_dofs(self):
        """
        Traverse the connected component tree and collect all DOFs associated with free
        coarse nodes also keep track of how many times a connected edge component appears
        (its multiplicity).

        The result is a dictionary very similar to the component tree, only now it looks like this:
        {
            coarse_node_i1: {
                "dof": int, # DOF of the coarse node
                coarse_edge_j1: [dof1, dof2, ...], # DOFs of the coarse edge
                coarse_edge_j2: {...}
                ...
                coarse_edge_jk: {...}
            },
            coarse_node_i2: {
                coarse_edge_j1: {...},
                ...
            },
        })
        """
        free_component_tree_dofs = {}
        edge_component_multiplicity = {}
        component_tree = self.two_mesh.connected_component_tree
        for coarse_node, coarse_edges in component_tree.items():
            coarse_node_dofs = list(self.fespace.GetDofNrs(coarse_node))
            if coarse_node_dofs in self.free_coarse_node_dofs:
                free_component_tree_dofs[coarse_node] = {"node": coarse_node_dofs}
            else:
                continue  # skip coarse nodes without free DOFs

            # treat coarse edges
            free_component_tree_dofs[coarse_node]["edges"] = {}
            for coarse_edge, edge_components in coarse_edges.items():
                # increment the multiplicity of the coarse edge component
                edge_component_multiplicity[coarse_edge] = (
                    edge_component_multiplicity.get(coarse_edge, 0) + 1
                )
                # save all DOFs of the edge component
                dofs = []
                for edge in edge_components["fine_edges"]:
                    edge_dofs = list(self.fespace.GetDofNrs(edge))
                    dofs.extend(edge_dofs)
                for vertex in edge_components["fine_vertices"]:
                    vertex_dofs = list(self.fespace.GetDofNrs(vertex))
                    dofs.extend(vertex_dofs)
                free_component_tree_dofs[coarse_node]["edges"][coarse_edge] = dofs
        return free_component_tree_dofs, edge_component_multiplicity

    def get_prolongation_operator(self):
        """
        Get the prolongation operator for the finite element space.

        Returns:
            ngs.ProlongationOperator: The prolongation operator for the finite element space.
        """
        prolongation_operator = self.fespace.Prolongation().Operator(1)
        for level_idx in range(1, self.two_mesh.refinement_levels):
            prolongation_operator = (
                self.fespace.Prolongation().Operator(level_idx + 1)
                @ prolongation_operator
            )
        return sp.csc_matrix(prolongation_operator.ToDense().NumPy())

    def get_gridfunc(self, vals):
        """
        Get the grid function representing the DOFs on the mesh.

        Args:
            vals (np.ndarray): The values to plot at each DOF.

        Returns:
            ngs.GridFunction: The grid function representing the DOFs on the mesh.
        """
        vals = np.asarray(vals, dtype=float).flatten()
        assert (
            len(vals) == self.fespace.ndof
        ), f"Length of vals ({len(vals)}) does not match number of DOFs ({self.fespace.ndof})"
        grid_function = ngs.GridFunction(self.fespace)
        grid_function.vec.FV().NumPy()[:] = vals
        return grid_function

    @property
    def u(self):
        """
        Get the trial function of the finite element space.

        Returns:
            ngs.TrialFunction: The trial function of the finite element space.
        """
        return self.fespace.TrialFunction()

    @property
    def v(self):
        """
        Get the test function of the finite element space.

        Returns:
            ngs.TestFunction: The test function of the finite element space.
        """
        return self.fespace.TestFunction()

    def __repr__(self):
        repr_str = "FE space DOFs:"
        repr_str += f"\n\ttotal DOFs: {self.total_dofs}"
        repr_str += f"\n\tfespace dimension: {'X'.join([str(d) for d in self.ptype.dimensions])}"
        repr_str += f"\n\tinterior DOFs: {self.num_interior_dofs}"
        repr_str += f"\n\tedge DOFs: {self.num_edge_dofs}"
        repr_str += f"\n\tcoarse Node DOFs: {self.num_coarse_node_dofs}"
        return repr_str

    def _print_domain_dofs(self):
        repr_str = "Domain DOFs:"
        for subdomain, data in self.domain_dofs.items():
            repr_str += f"\n\t{subdomain.nr}:"
            repr_str += f"\n\t\t#interior: {len(data['interior'])}"
            repr_str += f"\n\t\t#coarse_nodes: {data['coarse_nodes']}"
            repr_str += f"\n\t\t#edges: {data['edges']}"
            repr_str += f"\n\t\t#layer: {len(data['layer'])}"
        return repr_str


if __name__ == "__main__":
    """
    Example usage: Load a TwoLevelMesh and construct a FESpace on it.
    """
    lx, ly = 1.0, 1.0
    coarse_mesh_size = 0.4
    refinement_levels = 3
    layers = 1
    two_mesh = TwoLevelMesh.load(
        lx=lx,
        ly=ly,
        coarse_mesh_size=coarse_mesh_size,
        refinement_levels=refinement_levels,
        layers=layers,
    )
    ptype = ProblemType.DIFFUSION
    fespace = FESpace(two_mesh, [HomogeneousDirichlet(ptype)], ptype)
    fespace.calculate_dofs()
    print(fespace)
