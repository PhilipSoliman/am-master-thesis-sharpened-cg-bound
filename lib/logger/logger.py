import logging
from typing import Any, List, Optional

from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
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
            fp = kwargs.pop("fp", None)
            fp_str = ""
            if fp:
                fp_dir = fp.parent.name
                fp_name = fp.name
                fp_str = f": [bold magenta]{fp_dir}/{fp_name}[/bold magenta]"
            self._log(
                self.SUBSTEP,
                f"{emoji}\t{message}{fp_str}",
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
        record.levelname = (
            f"[{color}]{record.levelname:<{self.LEVEL_NAME_WIDTH}}[/{color}]"
        )
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
    MAX_TASKS = 10

    TASK_COLORS = {
        TaskID(9): "[green]",
        TaskID(8): "[blue]",
        TaskID(7): "[magenta]",
        TaskID(6): "[cyan]",
        TaskID(5): "[yellow]",
        TaskID(4): "[red]",
        TaskID(3): "[white]",
        TaskID(2): "[bright_green]",
        TaskID(1): "[bright_blue]",
        TaskID(0): "[bright_magenta]",
    }

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
        self._task_index: TaskID = TaskID(self.MAX_TASKS)

    def add_task(
        self,
        description: str,
        start: bool = True,
        total: Optional[float] = 100.0,
        completed: int = 0,
        visible: bool = True,
        **fields: Any,
    ) -> TaskID:
        with self._lock:
            task = Task(
                self._task_index,
                self.TASK_COLORS[self._task_index] + description,
                total,
                completed,
                visible=visible,
                fields=fields,
                _get_time=self.get_time,
                _lock=self._lock,
            )
            self._tasks[self._task_index] = task
            if start:
                self.start_task(self._task_index)
            new_task_index = self._task_index
            self._task_index = TaskID(int(self._task_index) - 1) # reverse order of tasks

            # resort tasks
            self._tasks = dict(sorted(self._tasks.items())) # ensure tasks are sorted by index
        self.refresh()
        return new_task_index

    def remove_task(self, task_id: TaskID) -> None:
        """Delete a task if it exists.

        Args:
            task_id (TaskID): A task ID.

        """
        with self._lock:
            del self._tasks[task_id]
            self._task_index = TaskID(int(self._task_index) + 1)  # increment index

    def progress_started(self) -> bool:
        return any(task.started for task in self._tasks.values())
