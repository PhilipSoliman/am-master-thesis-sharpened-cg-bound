import logging

from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


##########
# LOGGER #
##########
class CustomLogger(logging.Logger):
    SUBSTEP = 15  # Between INFO (20) and DEBUG (10)
    logging.addLevelName(SUBSTEP, "SUBSTEP")

    def substep(self, message, *args, emoji="↳", **kwargs):
        if self.isEnabledFor(self.SUBSTEP):
            self._log(
                self.SUBSTEP,
                f"{emoji}\t{message}",
                args,
                **kwargs,
            )


logging.setLoggerClass(CustomLogger)


class ColorLevelFormatter(logging.Formatter):
    LEVEL_NAME_WIDTH = 8
    INFO_COLOR = "blue"
    SUBSTEP_COLOR = "cyan"
    DEBUG_COLOR = "yellow"
    ERROR_COLOR = "red"

    def format(self, record):
        color = self.get_color(record)
        record.levelname = f"[{color}]{record.levelname:<{self.LEVEL_NAME_WIDTH}}[/{color}]"
        record.message = f"[{color}]{record.getMessage()}[/{color}]"
        return super().format(record)

    def get_color(self, record):
        if record.levelno == logging.INFO:
            return self.INFO_COLOR
        elif record.levelno == CustomLogger.SUBSTEP:
            return self.SUBSTEP_COLOR
        elif record.levelno == logging.DEBUG:
            return self.DEBUG_COLOR
        elif record.levelno == logging.ERROR:
            return self.ERROR_COLOR
        else:
            return "white"


handler = RichHandler(
    rich_tracebacks=True,
    markup=True,
    show_level=False,
    show_time=True,
    show_path=True,
)

handler.setFormatter(ColorLevelFormatter("%(levelname)s %(message)s"))

LOGGER: CustomLogger = logging.getLogger("lib")  # type: ignore[assignment]
LOGGER.setLevel("DEBUG")
LOGGER.addHandler(handler)


################
# PROGRESS BAR #
################
class AvgTimePerIterColumn(ProgressColumn):
    def render(self, task: TaskID) -> str:  # type: ignore
        if task.completed == 0:  # type: ignore
            return "—"
        avg = task.elapsed / task.completed  # type: ignore
        return f"{avg:.2f}s/it"


class PROGRESS(Progress):
    def __init__(self, *args, **kwargs):
        super().__init__(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            AvgTimePerIterColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            *args,
            **kwargs,
        )
