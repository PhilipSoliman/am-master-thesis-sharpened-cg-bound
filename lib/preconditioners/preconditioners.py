from types import NoneType
from typing import Callable, Optional, Type

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import LinearOperator, SuperLU, factorized
from tqdm import tqdm

from lib.fespace import FESpace
from lib.logger import LOGGER, PROGRESS
from lib.meshes import TwoLevelMesh
from lib.operators import Operator
from lib.preconditioners import CoarseSpace
from lib.solvers import DirectSparseSolver, MatrixType
from lib.utils import send_matrix_to_gpu, suppress_output


class OneLevelSchwarzPreconditioner(Operator):
    def __init__(
        self,
        A: sp.csr_matrix,
        fespace: FESpace,
        gpu_device: str | None = None,
        progress: Optional[PROGRESS] = None,
    ):
        self.progress = PROGRESS.get_active_progress_bar(progress)
        task = self.progress.add_task(
            "Initializing 1-level Schwarz preconditioner", total=3
        )
        LOGGER.info("Initializing 1-level Schwarz preconditioner")

        self.shape = A.shape
        self.fespace = fespace
        self.gpu_device = gpu_device
        self.local_free_dofs = self._get_local_free_dofs()
        self.progress.advance(task)
        self.local_operators = self._get_local_operators(A)
        self.progress.advance(task)
        self.local_solvers = self._get_local_solvers(A)
        self.progress.advance(task)
        self.name = "1-level Schwarz preconditioner"

        LOGGER.info("1-level Schwarz preconditioner initialized")
        self.progress.soft_stop()

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
        LOGGER.debug(f"Obtained local free dofs for {len(local_free_dofs)} subdomains")
        return local_free_dofs

    def _get_local_operators(self, A: sp.csr_matrix) -> list[sp.csc_matrix]:
        """Get local solvers for each subdomain."""
        local_operators = []
        task = self.progress.add_task(
            "Obtaining local operators", total=len(self.local_free_dofs)
        )
        for dofs in self.local_free_dofs:
            operator = A[dofs, :][:, dofs].tocsc()
            local_operators.append(operator)
            self.progress.advance(task)
        LOGGER.debug(f"Obtained local operators for {len(local_operators)} subdomains")
        self.progress.remove_task(task)
        return local_operators

    def _get_local_solvers(
        self, A: sp.csr_matrix
    ) -> list[
        Callable[[np.ndarray], np.ndarray]
        | Callable[[torch.Tensor, torch.Tensor], NoneType]
    ]:
        """Get local solvers for each subdomain."""
        local_solvers = []
        task = self.progress.add_task(
            "Obtaining local solvers", total=len(self.local_operators)
        )
        for operator in self.local_operators:
            if self.gpu_device is None:
                solver_f = factorized(operator)
            else:
                with suppress_output():
                    solver = DirectSparseSolver(
                        operator, matrix_type=MatrixType.SPD
                    ).solver
                solver_f = lambda rhs, out: solver.solve(rhs, out)  # type: ignore
            local_solvers.append(solver_f)
            self.progress.advance(task)
        LOGGER.debug(
            f"Obtained local solvers for {len(local_solvers)} subdomains on {'GPU' if self.gpu_device else 'CPU'}"
        )
        self.progress.remove_task(task)
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
        progress: Optional[PROGRESS] = None,
    ):
        super().__init__(A, fespace, gpu_device, progress)
        self.progress = PROGRESS.get_active_progress_bar(progress)
        task = self.progress.add_task(
            "Initializing 2-level Schwarz preconditioner",
            total=3 if gpu_device is None else 4,
        )

        # get coarse space
        self.coarse_space = coarse_space(A, fespace, two_mesh, progress=self.progress)
        self.progress.advance(task)

        # get coarse operator
        coarse_op = self.coarse_space.assemble_coarse_operator(A)
        self.progress.advance(task)
        if self.gpu_device is not None:
            self.restriction_operator = send_matrix_to_gpu(self.coarse_space.restriction_operator, self.gpu_device)  # type: ignore
            self.restriction_operator_T = send_matrix_to_gpu(self.coarse_space.restriction_operator.transpose(), self.gpu_device)  # type: ignore
            LOGGER.info("Restriction operators sent to GPU")
            self.progress.advance(task)

        # get coarse solver
        self.coarse_solver = self._get_coarse_solver(coarse_op)
        self.progress.advance(task)

        # set preconditioner name
        self.name = f"2-level Schwarz preconditioner with {self.coarse_space}"

        LOGGER.info(f"{self.name} initialized")
        self.progress.soft_stop()

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
        return x_1 + x_2

    def _get_coarse_solver(
        self,
        coarse_op: sp.csc_matrix,
    ) -> (
        Callable[[np.ndarray], np.ndarray]
        | Callable[[torch.Tensor, torch.Tensor], NoneType]
    ):
        if self.gpu_device is None:
            LOGGER.debug("Obtained coarse solver (CPU)")
            return factorized(coarse_op)
        else:
            with suppress_output():
                solver = DirectSparseSolver(
                    coarse_op, matrix_type=MatrixType.SPD
                ).solver
            solver_f = lambda rhs, out: solver.solve(rhs, out)  # type: ignore
            LOGGER.debug("Obtained coarse solver (GPU)")
            return solver_f

    def get_restriction_operator_bases(self) -> dict[str, ngs.GridFunction]:
        return self.coarse_space.get_restriction_operator_bases()
