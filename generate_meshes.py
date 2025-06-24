from lib.boundary_conditions import HomogeneousDirichlet
from lib.fespace import FESpace
from lib.logger import LOGGER, PROGRESS
from lib.meshes import TwoLevelMesh
from lib.problem_type import ProblemType

# set logger level
LOGGER.setLevel(LOGGER.INFO)

# mesh parameters
lx, ly = 1.0, 1.0
coarse_mesh_size_list = [1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64]
refinement_levels = 4
layers = 2

# progress bar setup
progress = PROGRESS.get_active_progress_bar()
task = progress.add_task("Generating mesh", total=len(coarse_mesh_size_list))
desc = progress.get_description(task) + " H = 1/{0:.0f}"

# main loop to generate meshes
for coarse_mesh_size in coarse_mesh_size_list:
    # update the description with the current mesh size
    progress.update(task, advance=0, description=desc.format(1 / coarse_mesh_size))

    # generate and save the two-level mesh
    two_mesh = TwoLevelMesh(
        lx,
        ly,
        coarse_mesh_size,
        refinement_levels=refinement_levels,
        layers=layers,
        progress=progress,
    )
    two_mesh.save()

    # generate and save the subdomain DOFs
    ptype = ProblemType.DIFFUSION
    boundary_conditions = [HomogeneousDirichlet(ptype)]
    fespace = FESpace(two_mesh, [HomogeneousDirichlet(ptype)], ptype, progress=progress)

    # increment the progress bar
    progress.advance(task)
