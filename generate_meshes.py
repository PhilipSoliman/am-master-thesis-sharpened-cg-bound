from lib.meshes import TwoLevelMesh
from lib.fespace import FESpace
from lib.problem_type import ProblemType
from lib.boundary_conditions import HomogeneousDirichlet
lx, ly = 1.0, 1.0
coarse_mesh_size_list = [1/4, 1/8, 1/16, 1/32, 1/64]
refinement_levels = 4
layers = 2

for coarse_mesh_size in coarse_mesh_size_list:
    print(f"Generating mesh & finite element space for coarse mesh size: {coarse_mesh_size}")

    # generate and save the two-level mesh
    two_mesh = TwoLevelMesh(lx, ly, coarse_mesh_size, refinement_levels=refinement_levels, layers=layers)
    two_mesh.save()

    # generate and save the subdomain DOFs
    ptype = ProblemType.DIFFUSION
    boundary_conditions = [HomogeneousDirichlet(ptype)]
    fespace = FESpace(two_mesh, [HomogeneousDirichlet(ptype)], ptype)
