from enum import Enum

import ngsolve as ngs


class ProblemType(Enum):
    """
    Enum for different types of problems.
    """

    DIFFUSION = ("diffusion", [ngs.H1], [1], [1])
    NAVIER_STOKES = ("navier_stokes", [ngs.VectorH1, ngs.H1], [3, 2], [2, 1])
    CUSTOM = ("custom", [], [], [])

    def __init__(self, title: str, fespaces: list, orders: list, dimensions: list):
        self._title = title
        self._fespaces = fespaces
        self._orders = orders
        self._dimensions = dimensions

    def __repr__(self):
        """
        Print the finite element spaces associated with the problem type.
        """
        repr_str = f"Finite Element Spaces for {self.title}:\n"
        for fespace, order, dim in zip(self.fespaces, self.orders, self.dimensions):
            repr_str += f"  - {fespace.__name__} (Order: {order}, Dimension: {dim})\n"
        return repr_str
    
    @property
    def title(self) -> str:
        """
        Get the title of the problem type.
        """
        return self.value
    
    @property
    def fespaces(self) -> list:
        """
        Get the finite element spaces associated with the problem type.
        """
        return self._fespaces
    
    @property
    def orders(self) -> list:
        """
        Get the orders of the finite element spaces associated with the problem type.
        """
        return self._orders
    
    @property
    def dimensions(self) -> list:
        """
        Get the dimensions of the finite element spaces associated with the problem type.
        """
        return self._dimensions
    
    @title.setter
    def title(self, value: str):
        """
        Set the title of the problem type.
        """
        self._title = value

    @fespaces.setter
    def fespaces(self, value: list):
        """
        Set the finite element spaces associated with the problem type.
        """
        self._fespaces = value

    @orders.setter
    def orders(self, value: list):
        """
        Set the orders of the finite element spaces associated with the problem type.
        """
        self._orders = value

    @dimensions.setter
    def dimensions(self, value: list):
        """
        Set the dimensions of the finite element spaces associated with the problem type.
        """
        self._dimensions = value