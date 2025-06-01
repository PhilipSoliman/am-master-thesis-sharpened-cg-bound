from fespace import FESpace
import scipy.sparse as sp
from coarse_space import CoarseSpace

class Preconditioner(object):
    def apply(self, x):
        raise NotImplementedError("This method should be implemented by subclasses.")

class OneLevelSchwarzPreconditioner(Preconditioner):
    def __init__(self, fespace: FESpace, A: sp.csr_matrix):
        self.fespace = fespace
        self.A = A

    def apply(self, x):
        # Implement the one-level Schwarz preconditioner application logic here
        pass

class TwoLevelSchwarzPreconditioner(OneLevelSchwarzPreconditioner):
    def __init__(self, fespace: FESpace, A: sp.csr_matrix, coarse_space: CoarseSpace):
        super().__init__(fespace, A)
        self.coarse_space = coarse_space

    def apply(self, x):
        x = super().apply(x)
        return self.coarse_space.apply(x)
