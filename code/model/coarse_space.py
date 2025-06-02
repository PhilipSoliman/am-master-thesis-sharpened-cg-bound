import numpy as np
import scipy.sparse as sp
from fespace import FESpace
from mesh import TwoLevelMesh


class CoarseSpace(object):
    def __init__(self, A: sp.csr_matrix, fespace: FESpace, two_mesh: TwoLevelMesh):
        self.fespace = fespace
        self.A = A
        self.two_mesh = two_mesh

    def apply(self, x: np.ndarray):
        raise NotImplementedError("This method should be implemented by subclasses.")


class AMSCoarseSpace(CoarseSpace):
    def apply(self, x: np.ndarray):
        # Implement the AMS coarse space application logic here
        raise NotImplementedError("AMS coarse space application not implemented.")


class GDSWCoarseSpace(CoarseSpace):
    def apply(self, x: np.ndarray):
        # Implement the GDSW coarse space application logic here
        raise NotImplementedError("GDSW coarse space application not implemented.")


class RGDSWCoarseSpace(CoarseSpace):
    def apply(self, x: np.ndarray):
        # Implement the RGDSW coarse space application logic here
        raise NotImplementedError("RGDSW coarse space application not implemented.")
