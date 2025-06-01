import numpy as np
import scipy.sparse as sp
from fespace import FESpace


class CoarseSpace(object):
    def apply(self, x: np.ndarray):
        raise NotImplementedError("This method should be implemented by subclasses.")


class AMSCoarseSpace(CoarseSpace):
    def __init__(self, fespace: FESpace, A: sp.csr_matrix):
        self.fespace = fespace
        self.A = A

    def apply(self, x: np.ndarray):
        # Implement the AMS coarse space application logic here
        raise NotImplementedError("AMS coarse space application not implemented.")


class GDSWCoarseSpace(CoarseSpace):
    def __init__(self, fespace: FESpace, A: sp.csr_matrix):
        self.fespace = fespace
        self.A = A

    def apply(self, x):
        # Implement the GDSW coarse space application logic here
        raise NotImplementedError("GDSW coarse space application not implemented.")


class RGDSWCoarseSpace(CoarseSpace):
    def __init__(self, fespace: FESpace, A: sp.csr_matrix):
        self.fespace = fespace
        self.A = A

    def apply(self, x: np.ndarray):
        # Implement the RGDSW coarse space application logic here
        raise NotImplementedError("RGDSW coarse space application not implemented.")
