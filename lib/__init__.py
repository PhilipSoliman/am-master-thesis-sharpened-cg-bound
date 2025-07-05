import os

from lib.logger import LOGGER, PROGRESS

try:
    # look for command line arguments to set the log level
    from lib.utils import get_cli_args

    args = get_cli_args()
    if hasattr(args, "loglvl"):
        LOGGER.setLevel(args.loglvl)
    if hasattr(args, "show_progress") and args.show_progress:
        LOGGER.debug("Showing progress bar (CLI argument)")
        PROGRESS.show()
    else:
        LOGGER.debug("Hiding progress bar (CLI argument)")
        PROGRESS.hide()
except Exception as e:
    LOGGER.exception(
        f"Failed to set log level or progress bar visibility from CLI arguments. Using default settings. {e}"
    )
    pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
