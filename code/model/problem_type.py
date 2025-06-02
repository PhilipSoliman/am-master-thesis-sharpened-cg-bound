from enum import Enum


class ProblemType(Enum):
    """
    Enum for different types of problems.
    """

    DIFFUSION = "diffusion"
    ADVECTION = "advection"
