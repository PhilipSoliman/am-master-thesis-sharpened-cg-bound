from lib.boundary_conditions import HomogeneousDirichlet
from lib.fespace import FESpace
from lib.logger import LOGGER, PROGRESS
from lib.meshes import DefaultMeshParams, TwoLevelMesh
from lib.problem_type import ProblemType

# set logger level
LOGGER.setLevel(LOGGER.INFO)

# progress bar setup
progress = PROGRESS.get_active_progress_bar()
task = progress.add_task("Generating mesh", total=len(DefaultMeshParams))
desc = progress.get_description(task) + " H = 1/{0:.0f}"

# main loop to generate meshes
for mesh_params in DefaultMeshParams:
    # update the description with the current mesh size
    progress.update(
        task, advance=0, description=desc.format(1 / mesh_params.coarse_mesh_size)
    )

    # generate and save the two-level mesh
    two_mesh = TwoLevelMesh(mesh_params, progress=progress)
    two_mesh.save()

    # generate and save the subdomain DOFs
    ptype = ProblemType.DIFFUSION
    boundary_conditions = [HomogeneousDirichlet(ptype)]
    fespace = FESpace(two_mesh, [HomogeneousDirichlet(ptype)], ptype, progress=progress)

    # increment the progress bar
    progress.advance(task)

progress.soft_stop()
