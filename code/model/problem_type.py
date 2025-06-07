from enum import Enum

import ngsolve as ngs


class ProblemType(Enum):
    """
    Enum for different types of problems.
    """

    DIFFUSION = ("diffusion", [ngs.H1], [1], [1])
    NAVIER_STOKES = ("navier_stokes", [ngs.VectorH1, ngs.H1], [3, 2], [2, 1])

    def __init__(self, title, fespaces, orders, dimensions):
        self.title = title
        self.fespaces = fespaces
        self.orders = orders
        self.dimensions = dimensions

    def __repr__(self):
        """
        Print the finite element spaces associated with the problem type.
        """
        repr_str = f"Finite Element Spaces for {self.title}:\n"
        for fespace, order, dim in zip(self.fespaces, self.orders, self.dimensions):
            repr_str += f"  - {fespace.__name__} (Order: {order}, Dimension: {dim})\n"
        return repr_str