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


class MatrixType(Enum):
    SPD = "SPD"
    Symmetric = "Symmetric"
    General = "General"



class DirectSparseSolver:
    NUM_CPU_THREADS = multiprocessing.cpu_count()
    GPU_BATCH_SIZE = 128
    CPU_BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, A: sp.csc_matrix, matrix_type: Optional[MatrixType] = None, multithreaded: bool = False):
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
        print("Initializing direct sparse solver")
        self.A = A
        self.matrix_type = matrix_type
        self.multithreaded = multithreaded
        self.solver = self.get_solver()

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
        print(f"\tgetting direct solver for {self.matrix_type} matrix")
        if self.matrix_type == MatrixType.SPD:
            print(f"\tusing cholesky solver")
            A_coo = self.A.tocoo()
            n = self.A.shape[0]  # type: ignore
            if self.DEVICE == "cuda":
                print(f"\tusing GPU device: {self.DEVICE}")
                self.batch_size = self.GPU_BATCH_SIZE
            else:
                print(f"\tusing CPU device: {self.DEVICE}")
                self.batch_size = self.CPU_BATCH_SIZE

            rows = torch.tensor(A_coo.row, dtype=torch.float64, device=self.DEVICE)
            cols = torch.tensor(A_coo.col, dtype=torch.float64, device=self.DEVICE)
            data = torch.tensor(A_coo.data, dtype=torch.float64, device=self.DEVICE)
            return chol.CholeskySolverD(n, rows, cols, data, chol.MatrixType.COO)  # type: ignore
        elif (
            self.matrix_type == MatrixType.Symmetric
            or self.matrix_type == MatrixType.General
        ):
            print(f"\tusing scipy factorized solver")
            return factorized(self.A)
        else:
            raise ValueError(
                "Direct solver is not available for non-symmetric matrices."
            )

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
            raise ValueError(
                f"Invalid matrix type: {matrix_type}. Must be None or an instance of MatrixType."
            )

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
        n_rows, n_rhs = rhs.shape  # type: ignore
        num_batches = (n_rhs + self.batch_size - 1) // self.batch_size
        out_cols = []
        for i in tqdm(
            range(num_batches),
            desc="Cholesky solving batches",
            unit="batch",
            total=num_batches,
        ):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, n_rhs)

            # rhs
            shape = (n_rows, end - start)
            rhs_batch = rhs[:, start:end].tocoo()
            rhs_batch_array = np.empty(shape, dtype=np.float64)
            rhs_batch_array[rhs_batch.row, rhs_batch.col] = rhs_batch.data
            rhs_device = torch.tensor(
                rhs_batch_array, dtype=torch.float64, device=self.DEVICE
            )

            # output
            x = np.zeros_like(rhs_batch_array)
            x_device = torch.tensor(x, dtype=torch.float64, device=self.DEVICE)

            # solve on GPU
            self.solver.solve(rhs_device, x_device)  # type: ignore

            # Move to CPU if necessary
            x = x_device.cpu()

            # Append to output columns
            out_cols.append(sp.csc_matrix(x.numpy()))  # type: ignore

        # Stack all columns into a sparse matrix
        return sp.hstack(out_cols).tocsc()  # type: ignore
    
    def lu(self, rhs: sp.csc_matrix) -> sp.csc_matrix:
        n_rhs = rhs.shape[1] # type: ignore
        cols = []
        for i in tqdm(
            range(n_rhs),
            desc="Solving interior operators",
            unit="operator",
            total=n_rhs,
        ):
            x = self.solver(rhs[:, i].toarray().ravel())
            cols.append(sp.csc_matrix(-x.reshape(-1, 1)))
        return sp.hstack(cols).tocsc() # type: ignore

    def lu_threaded(self, rhs: sp.csc_matrix) -> sp.csc_matrix:
        """
        Solve the system using LU decomposition on CPU, parallelized over columns.

        Args:
            rhs (sp.csc_matrix): The right-hand side matrix.

        Returns:
            sp.csc_matrix: The solved interior operator as a sparse matrix.
        """
        n_rows, n_rhs = rhs.shape  # type: ignore
        num_threads = self.NUM_CPU_THREADS
        chunk_size = (n_rhs + num_threads - 1) // num_threads

        def solve_chunk(col_range, pbar_update):
            results = []
            for i in col_range:
                x = self.solver(rhs[:, i].toarray().ravel())
                results.append((i, sp.csc_matrix(x.reshape(-1, 1))))
                pbar_update(1)
            return results

        # Prepare column ranges for each thread
        col_ranges = [
            range(i, min(i + chunk_size, n_rhs)) for i in range(0, n_rhs, chunk_size)
        ]

        results = []
        lock = threading.Lock()
        with tqdm(total=n_rhs, desc="LU solving columns", unit="col") as pbar:
            tqdm.set_lock(lock)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(solve_chunk, col_range, pbar.update) for col_range in col_ranges
                ]
                for future in concurrent.futures.as_completed(futures):
                    results.extend(future.result())

        # Sort results by column index
        results.sort(key=lambda tup: tup[0])
        sorted_cols = [col for idx, col in results]

        # Stack all columns into a sparse matrix in the original order
        return sp.hstack(sorted_cols).tocsc()  # type: ignore
