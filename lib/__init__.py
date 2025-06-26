import os

from lib.logger import LOGGER, PROGRESS

from .gpu_interface import GPUInterface

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# set default logging level for imported lib
LOGGER.setLevel("INFO")

# get GPU interface
gpu_interface = GPUInterface()
