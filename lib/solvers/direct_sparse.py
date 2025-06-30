import concurrent.futures
import multiprocessing
import threading
from enum import Enum
from typing import Callable, Optional

import cholespy as chol
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import eigsh, factorized, splu, spsolve
from tqdm import tqdm

from lib import gpu_interface as gpu
from lib.logger import LOGGER, PROGRESS


class MatrixType(Enum):
    SPD = "SPD"
    Symmetric = "Symmetric"
    General = "General"


class DirectSparseSolver:
    NUM_CPU_THREADS = multiprocessing.cpu_count()
    GPU_BATCH_SIZE = 128
    CPU_BATCH_SIZE = 32
    DEVICE = gpu.DEVICE

    def __init__(
        self,
        A: sp.csc_matrix,
        matrix_type: Optional[MatrixType] = None,
        multithreaded: bool = False,
        progress: Optional[PROGRESS] = None,
    ):
        """
        Initialize a sparse solver. For SPD matrices a sparse Cholesky decomposition is made and efficiently
        reused for solving thereafter. If GPU is available, the sparse solve is performed on the GPU, otherwise on the CPU.
        For symmetric and general matrices, a scipy factorized solver is used.

        Multithreading, if enabled, is only used for symmetric and general matrices.

        If the matrix type is not provided, it will be inferred from the matrix properties.

        This solver is most efficient for medium-sized (O(n^3)-O(n^4)), sparse SPD matrices.

        Args:
            A (sp.csc_matrix): A symmetric matrix.
            matrix_type (Optional[MatrixType]): The type of the matrix (SPD, Symmetric, General).
            multithreaded (bool): Whether to use multithreaded (only relevant for Symmetric and General matrices).
        """
        LOGGER.debug("Initializing direct sparse solver")
        self.A = A
        self.matrix_type = matrix_type
        self.multithreaded = multithreaded
        self._solver = self.get_solver()
        self.progress = progress
        LOGGER.debug("Initialized direct sparse solver")

    def __call__(
        self,
        rhs: sp.csc_matrix,
    ) -> sp.csc_matrix:
        """
        Multithreaded sparse solver for the interior operators.

        Args:
            A (sp.csc_matrix): A symmetric matrix.
            rhs (sp.csc_matrix): The right-hand side matrix.

        Returns:
            sp.csc_matrix: The solved interior operator as a sparse matrix.
        """
        if self.matrix_type == MatrixType.SPD:
            return self.cholesky(rhs)
        elif (
            self.matrix_type == MatrixType.Symmetric
            or self.matrix_type == MatrixType.General
        ):
            if self.multithreaded:
                return self.lu_threaded(rhs)
            else:
                return self.lu(rhs)
        else:
            raise ValueError(
                f"No direct solver available for matrix type {self.matrix_type}."
            )

    def get_solver(self) -> chol.CholeskySolverD | Callable[[np.ndarray], np.ndarray]:  # type: ignore
        LOGGER.debug(f"getting direct solver for {self.matrix_type} matrix")

        # select solver based on matrix type
        if self.matrix_type == MatrixType.SPD:
            LOGGER.debug(f"using cholesky solver")

            if gpu.AVAILABLE:
                self.batch_size = self.GPU_BATCH_SIZE
            else:
                self.batch_size = self.CPU_BATCH_SIZE
            n = self.A.shape[0]  # type: ignore
            A_coo = self.A.tocoo()
            rows = torch.tensor(A_coo.row, dtype=torch.float64, device=self.DEVICE)
            cols = torch.tensor(A_coo.col, dtype=torch.float64, device=self.DEVICE)
            data = torch.tensor(A_coo.data, dtype=torch.float64, device=self.DEVICE)
            return chol.CholeskySolverD(n, rows, cols, data, chol.MatrixType.COO)  # type: ignore
        elif (
            self.matrix_type == MatrixType.Symmetric
            or self.matrix_type == MatrixType.General
        ):
            LOGGER.debug(f"using scipy factorized solver")
            return factorized(self.A)
        else:
            msg = f"Unsupported matrix type: {self.matrix_type}"
            LOGGER.error(msg)
            raise ValueError(msg)

    @property
    def solver(self) -> Callable[[np.ndarray], np.ndarray] | chol.CholeskySolverD:  # type: ignore
        return self._solver

    @property
    def matrix_type(self) -> MatrixType:
        return self._matrix_type

    @matrix_type.setter
    def matrix_type(self, matrix_type: Optional[MatrixType]) -> None:
        if matrix_type is None:
            if self.is_symmetric():
                if self.is_spd():
                    self._matrix_type = MatrixType.SPD
                else:
                    self._matrix_type = MatrixType.Symmetric
            else:
                self._matrix_type = MatrixType.General
        elif isinstance(matrix_type, MatrixType):
            self._matrix_type = matrix_type
        else:
            msg = f"Invalid matrix type: {matrix_type}. Must be None or an instance of MatrixType."
            LOGGER.error(msg)
            raise ValueError(msg)

    def is_symmetric(self, tol=1e-10) -> bool:
        # For sparse matrices, compare nonzero structure and values
        return (self.A - self.A.T).nnz == 0 and np.allclose(
            self.A.data, self.A.T.data, atol=tol
        )

    def is_spd(self, tol=1e-10) -> bool:
        try:
            # Compute the smallest algebraic eigenvalue
            vals = eigsh(self.A, k=1, which="SA", return_eigenvectors=False)
            return vals[0] > tol  # type: ignore
        except Exception:  # eigsh requires symmetric input
            return False

    def cholesky(self, rhs: sp.csc_matrix) -> sp.csc_matrix:
        """
        Solve the system using Cholesky decomposition on GPU/CPU.

        Args:
            rhs (sp.csc_matrix): The right-hand side matrix.

        Returns:
            sp.csc_matrix: The solved interior operator as a sparse matrix.
        """
        progress = PROGRESS.get_active_progress_bar(self.progress)
        LOGGER.debug("Solving using Cholesky decomposition")
        n_rows, n_rhs = rhs.shape  # type: ignore
        num_batches = (n_rhs + self.batch_size - 1) // self.batch_size
        out_cols = []
        task = progress.add_task("Cholesky solving batches", total=num_batches)
        for i in range(num_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, n_rhs)

            # rhs
            shape = (n_rows, end - start)
            rhs_batch = rhs[:, start:end].tocoo()
            rhs_batch_array = np.zeros(shape, dtype=np.float64)
            rhs_batch_array[rhs_batch.row, rhs_batch.col] = rhs_batch.data
            rhs_device = gpu.send_array(rhs_batch_array, dtype=torch.float64)

            # output
            x = np.zeros_like(rhs_batch_array)
            x_device = gpu.send_array(x, dtype=torch.float64)

            # solve on GPU (if available) or CPU
            self.solver.solve(rhs_device, x_device)  # type: ignore

            # Move to CPU if necessary
            x = gpu.retrieve_array(x_device)

            # Append to output columns
            out_cols.append(sp.csc_matrix(x))  # type: ignore

            progress.advance(task)

        # Stack all columns into a sparse matrix
        out = sp.hstack(out_cols).tocsc()
        LOGGER.debug("Cholesky solving completed")
        progress.soft_stop()

        return out  # type: ignore

    def lu(self, rhs: sp.csc_matrix) -> sp.csc_matrix:
        """
        Solve the system using LU decomposition and single loop over rhs columns.

        This is efficient, as the factorized solver is used to solve each column of the right-hand side matrix
        independently. This is suitable for medium-sized matrices where the factorization is not too expensive.

        Note, this method instantiates an empty csc_matrix of the same shape as rhs and fills it with the solution
        for each column of rhs. This is is not as fast as saving the columns (possibly in batches) and stacking them
        at the end, but it is more memory efficient for large rhs matrices.
        """
        progress = PROGRESS.get_active_progress_bar(self.progress)
        LOGGER.debug("Solving using LU decomposition")
        n_rhs = rhs.shape[1]  # type: ignore
        out = sp.csc_matrix(rhs.shape)
        task = progress.add_task("LU solving columns", total=n_rhs)
        for i in range(n_rhs):
            x = self.solver(rhs[:, i].toarray().ravel())
            out[:, i] = sp.csc_matrix(x[:, None])
            progress.advance(task)

        LOGGER.debug("LU solving completed")
        progress.soft_stop()
        return out.tocsc()  # type: ignore

    def lu_threaded(self, rhs: sp.csc_matrix) -> sp.csc_matrix:
        """
        Solve the system using LU decomposition on CPU, multithreaded over rhs columns.

        Args:
            rhs (sp.csc_matrix): The right-hand side matrix.

        Returns:
            sp.csc_matrix: The solved interior operator as a sparse matrix.
        """
        progress = PROGRESS.get_active_progress_bar(self.progress)
        LOGGER.debug("Solving using LU decomposition with multithreading")
        n_rhs = rhs.shape[1]  # type: ignore
        num_threads = self.NUM_CPU_THREADS
        chunk_size = (n_rhs + num_threads - 1) // num_threads

        task = progress.add_task("LU solving columns", total=n_rhs)

        def solve_chunk(col_range, advance_task=lambda: progress.advance(task)):
            results = []
            for i in col_range:
                x = self.solver(rhs[:, i].toarray().ravel())
                results.append((i, sp.csc_matrix(x.reshape(-1, 1))))
                advance_task()
            return results

        # Prepare column ranges for each thread
        col_ranges = [
            range(i, min(i + chunk_size, n_rhs)) for i in range(0, n_rhs, chunk_size)
        ]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(solve_chunk, col_range) for col_range in col_ranges
            ]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())

        # Sort results by column index
        results.sort(key=lambda tup: tup[0])
        sorted_cols = [col for idx, col in results]

        # Stack all columns into a sparse matrix in the original order
        out = sp.hstack(sorted_cols).tocsc()
        LOGGER.debug("LU solving with multithreading completed")
        progress.soft_stop()

        return out  # type: ignore
