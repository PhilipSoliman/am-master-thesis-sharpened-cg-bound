import ngsolve as ngs
from mesh import TwoLevelMesh


class FESpace:
    """
    FESpace
    A class for constructing and managing a finite element space on a TwoLevelMesh,
    including the classification of DOFs (degrees of freedom) into interior, edge, and coarse node categories.
    """

    def __init__(
        self, two_mesh: TwoLevelMesh, order: int = 1, discontinuous: bool = False, **bcs
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
        if discontinuous:
            self.fespace = ngs.H1(two_mesh.fine_mesh, order=order, discontinuous=True, **bcs)
        else:
            self.fespace = ngs.H1(two_mesh.fine_mesh, order=order, **bcs)

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
        # get fine degrees of freedom (DOFs)
        interior = set()
        for el in self.fespace.Elements():
            dofs = self.fespace.GetDofNrs(el)
            interior.update(dofs)

        # calculate subdomain DOFs
        self.calculate_subdomain_dofs()

        # remove coarse node DOFs from fine DOFs
        coarse_node_dofs = set()
        edge_dofs = set()
        for domain, subdomain_data in self.domain_dofs.items():
            edge_dofs.update(subdomain_data["edges"])
            coarse_node_dofs.update(subdomain_data["coarse_nodes"])

        # remove edge DOFs from interio DOFs
        interior -= edge_dofs
        interior -= coarse_node_dofs

        # save DOFS
        self.interior_dofs = list(interior)
        self.num_interior_dofs = len(self.interior_dofs)
        self.edge_dofs = list(edge_dofs)
        self.num_edge_dofs = len(self.edge_dofs)
        self.coarse_node_dofs = list(coarse_node_dofs)
        self.num_coarse_node_dofs = len(self.coarse_node_dofs)

        print("FE space DOFS:")
        print(f"\t#total: {self.fespace.ndof}")
        print(f"\t#interior: {self.num_interior_dofs}")
        print(f"\t#edge: {self.num_edge_dofs}")
        print(f"\t#coarse_node: {len(coarse_node_dofs)}")

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

            subdomain_edge_dofs = set()
            for e in subdomain_data["edges"]:
                v1, v2 = self.two_mesh.fine_mesh.edges[e.nr].vertices

                # get vertex DOFs
                subdomain_edge_dofs.update(self.fespace.GetDofNrs(v1))
                subdomain_edge_dofs.update(self.fespace.GetDofNrs(v2))

                # get edge subdomain_edge_dofs
                subdomain_edge_dofs.update(self.fespace.GetDofNrs(e))
            subdomain_edge_dofs -= coarse_node_dofs
            interior_dofs -= subdomain_edge_dofs

            layer_dofs = set()
            for layer_idx in range(1, self.two_mesh.layers + 1):
                layer_elements = subdomain_data[f"layer_{layer_idx}"]
                for el in layer_elements:
                    dofs = self.fespace.GetDofNrs(el)
                    layer_dofs.update(dofs)
            layer_dofs -= subdomain_edge_dofs
            layer_dofs -= coarse_node_dofs
            layer_dofs -= interior_dofs

            self.domain_dofs[subdomain] = {
                "interior": list(interior_dofs),
                "coarse_nodes": list(coarse_node_dofs),
                "edges": list(subdomain_edge_dofs),
                "layer": list(layer_dofs),
            }
        return self.domain_dofs

    @property
    def u (self):
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

if __name__ == "__main__":
    """
    Example usage: Load a TwoLevelMesh and construct a FESpace on it.
    """
    lx, ly = 1.0, 1.0
    coarse_mesh_size = 0.15
    refinement_levels = 2
    layers= 2
    two_mesh = TwoLevelMesh.load(
        lx=lx,
        ly=ly,
        coarse_mesh_size=coarse_mesh_size,
        refinement_levels=refinement_levels,
        layers=layers
    )
    fespace = FESpace(two_mesh, order=1, discontinuous=False)
