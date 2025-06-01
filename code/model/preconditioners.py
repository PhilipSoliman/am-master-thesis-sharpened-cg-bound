import numpy as np
import scipy.sparse as sp
from coarse_space import CoarseSpace
from mesh import TwoLevelMesh


class Preconditioner(object):
    def apply(self, x: np.ndarray):
        raise NotImplementedError("This method should be implemented by subclasses.")


class OneLevelSchwarzPreconditioner(Preconditioner):
    def __init__(self, A: sp.csr_matrix, two_mesh: TwoLevelMesh):
        self.two_mesh = two_mesh
        self.A = A

    def apply(self, x: np.ndarray):
        # Implement the one-level Schwarz preconditioner application logic here
        raise NotImplementedError(
            "One-level Schwarz preconditioner application not implemented."
        )


class TwoLevelSchwarzPreconditioner(OneLevelSchwarzPreconditioner):
    def __init__(self, A: sp.csr_matrix, two_mesh: TwoLevelMesh, coarse_space: CoarseSpace):
        super().__init__(A, two_mesh)
        self.coarse_space = coarse_space

    def apply(self, x: np.ndarray):
        x = super().apply(x)
        return self.coarse_space.apply(x)
