from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch as tr

from lib.logger import LOGGER


class GPUInterface:
    """
    Interface for sending matrices to the GPU.
    """

    DEVICE = "cuda" if tr.cuda.is_available() else "cpu"
    AVAILABLE = False
    AVAILABILTY_MSG = (
        "CUDA is not available. Please check if [link=https://developer.nvidia.com/cuda-downloads]CUDA[/link] is installed and configured correctly."
        "[link=https://pytorch.org/get-started/locally/]PyTorch CUDA installation guide[/link]"
    )

    def __init__(self):
        if not self.available():
            LOGGER.warning(self.AVAILABILTY_MSG)
        self.AVAILABLE = True
        LOGGER.debug(f"GPU Interface initialized. Using device: {self.DEVICE}")

    def available(self) -> bool:
        """
        Check if CUDA is available.

        Returns:
            bool: True if CUDA is available, False otherwise.
        """
        if self.DEVICE == "cpu":
            return False
        return True

    @staticmethod
    def send_matrix(
        mat: sp.csc_matrix | sp.csr_matrix, device: str, dense: bool = False
    ) -> tr.Tensor:
        """
        Send a sparse matrix to the GPU as a sparse COO tensor or dense tensor.

        Args:
            mat (sp.csc_matrix | sp.csr_matrix): The sparse matrix to send.
            device (str): The device to send the matrix to (e.g., 'cuda:0').
            dense (bool): If True, convert the sparse matrix to a dense tensor.

        Returns:
            tr.Tensor: The matrix on the specified device.
        """
        if not dense:
            coo = mat.tocoo()
            return tr.sparse_coo_tensor(np.array([coo.row, coo.col]), coo.data, coo.shape, device=device)  # type: ignore
        else:
            return tr.tensor(mat.toarray(), dtype=tr.float64, device=device)  # type: ignore

    @staticmethod
    def send_array(
        arr: np.ndarray, device: str = DEVICE, dtype: Optional[tr.dtype] = None
    ) -> tr.Tensor:
        """
        Send a NumPy array to the GPU as a tensor.

        Args:
            arr (np.ndarray): The array to send.
            device (str): The device to send the array to (default is 'cuda').
            dtype (Optional[tr.dtype]): The data type of the tensor (default is None).

        Returns:
            tr.Tensor: The array as a tensor on the specified device.
        """
        return tr.tensor(arr, dtype=dtype, device=device)

    @staticmethod
    def retrieve_array(tensor: tr.Tensor) -> np.ndarray:
        """
        Retrieve a tensor from the GPU and convert it to a NumPy array.

        Args:
            tensor (tr.Tensor): The tensor to retrieve.
            device (str): The device to move the tensor to (default is 'cpu').

        Returns:
            np.ndarray: The tensor as a NumPy array.
        """
        return tensor.cpu().numpy()


if __name__ == "__main__":
    gpu_interface = GPUInterface()
