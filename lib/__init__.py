from .meshes import *
from .fespace import *
from .preconditioners import *
from .problems import *
from .solvers import *
# from utils import * we do not want to import everything from utils (better to keep it nested)

# __all__ = [
#     *meshes.__all__,
#     *fespace.__all__,
#     *preconditioners.__all__,
#     *problems.__all__,
#     *solvers.__all__,
# ]

# how should I change my setup.py script to reflect all the changes in the lib folder