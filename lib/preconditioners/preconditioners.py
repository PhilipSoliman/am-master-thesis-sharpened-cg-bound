from types import NoneType
from typing import Callable, Type

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import LinearOperator, SuperLU, factorized
from tqdm import tqdm

from lib.fespace import FESpace
from lib.meshes import TwoLevelMesh
from lib.operators import Operator
from lib.preconditioners import CoarseSpace
from lib.solvers import DirectSparseSolver, MatrixType
from lib.utils import send_matrix_to_gpu


class OneLevelSchwarzPreconditioner(Operator):
    def __init__(
        self, A: sp.csr_matrix, fespace: FESpace, gpu_device: str | None = None
    ):
        print("Initializing 1-level Schwarz preconditioner")
        self.shape = A.shape
        self.fespace = fespace
        self.gpu_device = gpu_device
        self.local_free_dofs = self._get_local_free_dofs()
        self.local_operators = self._get_local_operators(A)
        self.local_solvers = self._get_local_solvers(A)

        self.name = "1-level Schwarz preconditioner"

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        for dofs, solver in zip(self.local_free_dofs, self.local_solvers):
            out[dofs] += solver(x[dofs])  # type: ignore
        return out

    def apply_gpu(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x, dtype=torch.float64)
        for dofs, solver in zip(self.local_free_dofs, self.local_solvers):
            tmp = torch.zeros_like(x[dofs], dtype=torch.float64)
            solver(x[dofs], tmp)  # type: ignore
            out[dofs] += tmp
            # del tmp
            # torch.cuda.empty_cache()  # Clear GPU memory
        return out

    def as_linear_operator(self) -> LinearOperator:
        """Return the preconditioner as a linear operator."""
        return LinearOperator(self.shape, lambda x: self.apply(x))

    def _get_local_free_dofs(self) -> list[np.ndarray]:
        """Get local free dofs for each subdomain."""
        local_free_dofs = []
        for subdomain_dofs in self.fespace.domain_dofs.values():
            local_free_dofs.append(
                self.fespace.map_global_to_restricted_dofs(np.array(subdomain_dofs))
            )
        return local_free_dofs

    def _get_local_operators(self, A: sp.csr_matrix) -> list[sp.csc_matrix]:
        """Get local solvers for each subdomain."""
        local_operators = []
        for dofs in self.local_free_dofs:
            operator = A[dofs, :][:, dofs].tocsc()
            local_operators.append(operator)
        return local_operators

    def _get_local_solvers(
        self, A: sp.csr_matrix
    ) -> list[
        Callable[[np.ndarray], np.ndarray]
        | Callable[[torch.Tensor, torch.Tensor], NoneType]
    ]:
        """Get local solvers for each subdomain."""
        local_solvers = []
        for operator in tqdm(
            self.local_operators,
            desc="Getting local solvers",
            total=len(self.local_operators),
        ):
            if self.gpu_device is None:
                solver_f = factorized(operator)
            else:
                solver = DirectSparseSolver(operator, matrix_type=MatrixType.SPD).solver
                solver_f = lambda rhs, out: solver.solve(rhs, out)  # type: ignore
            local_solvers.append(solver_f)
        return local_solvers

    def __str__(self):
        return self.name


class TwoLevelSchwarzPreconditioner(OneLevelSchwarzPreconditioner):
    def __init__(
        self,
        A: sp.csr_matrix,
        fespace: FESpace,
        two_mesh: TwoLevelMesh,
        coarse_space: Type[CoarseSpace],
        gpu_device: str | None = None,
    ):
        super().__init__(A, fespace, gpu_device)
        self.coarse_space = coarse_space(A, fespace, two_mesh)
        self.name = f"2-level Schwarz preconditioner with {self.coarse_space}"
        coarse_op = self.coarse_space.assemble_coarse_operator(A)
        if self.gpu_device is not None:
            self.restriction_operator = send_matrix_to_gpu(self.coarse_space.restriction_operator, self.gpu_device)  # type: ignore
            self.restriction_operator_T = send_matrix_to_gpu(self.coarse_space.restriction_operator.transpose(), self.gpu_device)  # type: ignore
        print("\tobtaining coarse solver")
        self.coarse_solver = self._get_coarse_solver(coarse_op)
        print("\tdone obtaining coarse solver")

    def apply(self, x: np.ndarray) -> np.ndarray:
        x_1 = super().apply(x)
        x_2 = self.coarse_space.restriction_operator.transpose() @ x
        x_2 = self.coarse_solver(x_2)  # type: ignore
        x_2 = self.coarse_space.restriction_operator @ x_2

        return x_1 + x_2

    def apply_gpu(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = super().apply_gpu(x)
        x_2 = torch.mv(self.restriction_operator_T, x)
        tmp = torch.zeros_like(x_2, dtype=torch.float64)
        self.coarse_solver(x_2, tmp)  # type: ignore
        x_2 = torch.mv(self.restriction_operator, tmp)  # type: ignore
        # del tmp
        return x_1 + x_2

    def _get_coarse_solver(
        self,
        coarse_op: sp.csc_matrix,
    ) -> (
        Callable[[np.ndarray], np.ndarray]
        | Callable[[torch.Tensor, torch.Tensor], NoneType]
    ):
        if self.gpu_device is None:
            return factorized(coarse_op)
        else:
            solver = DirectSparseSolver(coarse_op, matrix_type=MatrixType.SPD).solver
            solver_f = lambda rhs, out: solver.solve(rhs, out)  # type: ignore
            return solver_f

    def get_restriction_operator_bases(self) -> dict[str, ngs.GridFunction]:
        return self.coarse_space.get_restriction_operator_bases()
