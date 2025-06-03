import ngsolve as ngs
from mesh import TwoLevelMesh


class FESpace:
    """
    FESpace
    A class for constructing and managing a finite element space on a TwoLevelMesh,
    including the classification of DOFs (degrees of freedom) into interior, edge, and coarse node categories.
    """

    def __init__(
        self,
        two_mesh: TwoLevelMesh,
        order: int = 1,
        discontinuous: bool = False,
        dim: int = 1,
        **bcs,
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
        self.dimension = dim
        if dim == 1:
            self.fespace = ngs.H1(
                two_mesh.fine_mesh, order=order, **bcs
            )
        elif dim == 2:
            self.fespace = ngs.VectorH1(
                two_mesh.fine_mesh, order=order, **bcs
            )
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
        # get all degrees of freedom (DOFs)
        self.interior_dofs = set()
        for el in self.fespace.Elements():
            dofs = self.fespace.GetDofNrs(el)
            self.interior_dofs.update(dofs)

        # calculate subdomain DOFs
        self.calculate_subdomain_dofs()

        # remove coarse node DOFs from fine DOFs
        self.coarse_node_dofs = set()
        self.edge_dofs = set()
        for subdomain_data in self.domain_dofs.values():
            self.coarse_node_dofs.update(subdomain_data["coarse_nodes"])
            for coarse_edge_dofs in subdomain_data["edges"].values():
                self.edge_dofs.update(coarse_edge_dofs)

        # remove edge DOFs from interior DOFs
        self.interior_dofs -= self.edge_dofs
        self.interior_dofs -= self.coarse_node_dofs

        # save DOFS
        self.num_interior_dofs = len(self.interior_dofs)
        self.num_edge_dofs = len(self.edge_dofs)
        self.num_coarse_node_dofs = len(self.coarse_node_dofs)
        self.num_face_dofs = 0  # Not used in this implementation

        print("FE space DOFS:")
        print(f"\t#total: {self.fespace.ndof}")
        print(f"\t#interior: {self.num_interior_dofs}")
        print(f"\t#edge: {self.num_edge_dofs}")
        print(f"\t#coarse_node: {self.num_coarse_node_dofs}")

    def calculate_subdomain_dofs(self):
        """
        Calculate DOFs for each subdomain (coarse element) and classify them as interior, edge, coarse node, or layer DOFs.

        Returns:
            dict: Mapping from coarse elements to their classified DOFs.
        """
        self.domain_dofs = {}
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
            for coarse_edge_nr, fine_edges in subdomain_data["edges"].items():
                coarse_edge_dofs = set()
                for e in fine_edges:
                    v1, v2 = self.two_mesh.fine_mesh.edges[e.nr].vertices

                    # get vertex DOFs
                    coarse_edge_dofs.update(self.fespace.GetDofNrs(v1))
                    coarse_edge_dofs.update(self.fespace.GetDofNrs(v2))

                    # get edge subdomain_edge_dofs
                    coarse_edge_dofs.update(self.fespace.GetDofNrs(e))
                coarse_edge_dofs -= coarse_node_dofs
                interior_dofs -= coarse_edge_dofs
                subdomain_edge_dofs[coarse_edge_nr] = list(coarse_edge_dofs)
                all_subdomain_edge_dofs.update(coarse_edge_dofs)

            layer_dofs = set()
            for layer_idx in range(1, self.two_mesh.layers + 1):
                layer_elements = subdomain_data[f"layer_{layer_idx}"]
                for el in layer_elements:
                    dofs = self.fespace.GetDofNrs(el)
                    layer_dofs.update(dofs)
            layer_dofs -= all_subdomain_edge_dofs
            layer_dofs -= coarse_node_dofs
            layer_dofs -= interior_dofs

            self.domain_dofs[subdomain] = {
                "interior": list(interior_dofs),
                "coarse_nodes": list(coarse_node_dofs),
                "edges": subdomain_edge_dofs,
                "layer": list(layer_dofs),
            }
        return self.domain_dofs

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
        # print domain DOFs
        repr_str = "Domain DOFs:\n"
        for subdomain, data in self.domain_dofs.items():
            repr_str += f"\t{subdomain.nr}:\n"
            repr_str += f"\t\t#interior: {len(data['interior'])}\n"
            repr_str += f"\t\t#coarse_nodes: {len(data['coarse_nodes'])}\n"
            repr_str += f"\t\t#edges: {data['edges']}\n"
            repr_str += f"\t\t#layer: {len(data['layer'])}\n"
        return repr_str

if __name__ == "__main__":
    """
    Example usage: Load a TwoLevelMesh and construct a FESpace on it.
    """
    lx, ly = 1.0, 1.0
    coarse_mesh_size = 0.15
    refinement_levels = 2
    layers = 2
    two_mesh = TwoLevelMesh.load(
        lx=lx,
        ly=ly,
        coarse_mesh_size=coarse_mesh_size,
        refinement_levels=refinement_levels,
        layers=layers,
    )
    fespace = FESpace(two_mesh, order=1, discontinuous=False)
    print(fespace)
