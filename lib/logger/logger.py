import logging
from pathlib import Path
from typing import Any, List, Optional

from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
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
# Helper function to format file paths for logging
def format_fp(fp: Optional[Path]) -> str:
    """Format a file path for logging."""
    if fp is None:
        return ""
    fp_dir = fp.parent.name
    fp_name = fp.name
    return f"{fp_dir}/{fp_name}"


# Custom logger class to handle Path formatting and custom log levels
class CustomLogger(logging.Logger):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
    ) -> None:
        # find Paths in args and format them
        args_new = []
        for arg in args:
            if isinstance(arg, Path):
                arg = format_fp(arg)
            args_new.append(arg)
        super()._log(
            level,
            msg,
            tuple(args_new),
            exc_info=exc_info,
            extra=extra,
            stack_info=stack_info,
            stacklevel=stacklevel,
        )


# Custom formatter to add colors to log levels
class ColorLevelFormatter(logging.Formatter):
    LEVEL_NAME_WIDTH = 8
    ERROR_COLOR = "red"
    WARNING_COLOR = "yellow"
    INFO_COLOR = "green"
    DEBUG_COLOR = "blue"
    LEN_TAB = 4
    TAB_FORMAT = f"|{'-'* (LEN_TAB - 2)}>"

    def format(self, record) -> str:
        record.levelcolor = self.get_color(record)
        emoji = self.get_emoji(record)
        record.levelname = f"{(emoji + record.levelname):<{self.LEVEL_NAME_WIDTH-1}}"
        prefix = self.get_prefix(record)
        message = record.getMessage()
        lines = message.splitlines()
        formatted_message = ""
        if len(lines) > 0:
            formatted_message = f"{prefix}{lines[0]}"
            if len(lines) > 1:
                beginning = f"\n{'':<{self.LEVEL_NAME_WIDTH+1}}{'':<{len(prefix)}}"
                for line in lines[1:]:
                    line = line.replace("\t", self.TAB_FORMAT)
                    line = f"{beginning}{line}"
                    formatted_message += line
        record.message = formatted_message
        return self.formatMessage(record)

    def get_color(self, record):
        if record.levelno == logging.ERROR:
            return self.ERROR_COLOR
        elif record.levelno == logging.INFO:
            return self.INFO_COLOR
        elif record.levelno == logging.DEBUG:
            return self.DEBUG_COLOR
        else:
            return "white"

    def get_emoji(self, record):
        """Get the emoji for the log record."""
        if record.levelno == logging.ERROR:
            return "❌ "  # exclamation mark emoji
        elif record.levelno == logging.WARNING:
            return "⚠️ "  # warning emoji
        elif record.levelno == logging.INFO:
            return "✅ "  # information emoji simple
        # elif record.levelno == logging.SUBSTEP:
        #     return "[cyan]"
        elif record.levelno == logging.DEBUG:
            return "🧿 "  # bug emoji
        else:
            return ""

    def get_prefix(self, record):
        """Get the prefix for the log record."""
        if record.levelno == logging.DEBUG:
            return f"➥{'':<{self.LEN_TAB}}"
        else:
            return ""


# instantiate the custom logger
logging.setLoggerClass(CustomLogger)
LOGGER: CustomLogger = logging.getLogger("lib")  # type: ignore[assignment]
LOGGER = CustomLogger("lib")  # type: ignore[assignment]
LOGGER.setLevel("DEBUG")

# set rich text handler
handler = RichHandler(
    rich_tracebacks=True,
    markup=True,
    show_level=False,
    show_time=True,
    show_path=True,
    log_time_format="[%Y-%m-%d %H:%M:%S.%f]",
)

handler.setFormatter(
    ColorLevelFormatter("[%(levelcolor)s]%(levelname)s %(message)s[/%(levelcolor)s]")
)
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
    MAX_TASKS = 9

    TASK_COLORS = {
        9: "[green]",
        8: "[blue]",
        7: "[magenta]",
        6: "[cyan]",
        5: "[yellow]",
        4: "[red]",
        3: "[white]",
        2: "[bright_green]",
        1: "[bright_blue]",
        0: "[bright_magenta]",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            AvgTimePerIterColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            *args,
            **kwargs,
        )
        self.already_active: bool = False
        self._task_index: TaskID = TaskID(self.MAX_TASKS)

    @classmethod
    def get_active_progress_bar(
        cls, progress: Optional["PROGRESS"] = None
    ) -> "PROGRESS":
        if isinstance(progress, PROGRESS):
            progress_already_active = True
            if not progress.progress_started():
                progress_already_active = False
                progress.start()
            setattr(progress, "already_active", progress_already_active)
            return progress
        elif progress is None:
            progress_already_active = False
            obj = cls.__new__(cls)
            obj.__init__()
            setattr(obj, "already_active", progress_already_active)
            obj.start()
            return obj

    def soft_start(self) -> None:
        """Only start progress if it was not already active when get_active_progress_bar was called."""
        if not self.progress_started():
            self.start()

    def soft_stop(self) -> None:
        """Only stop progress if it was not already active when get_active_progress_bar was called."""
        self.remove_task(TaskID(self._task_index + 1))
        if not self.already_active:
            self.stop()
            # print('\r', end='')  # Move cursor to the beginning of the line

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
            fp = fields.pop("fp", None)
            fp_str = format_fp(fp)
            desc = self.TASK_COLORS[int(self._task_index)] + description + fp_str
            task = Task(
                self._task_index,
                desc,
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
            self._task_index = TaskID(
                int(self._task_index) - 1
            )  # reverse order of tasks

            # resort tasks
            self._tasks = dict(
                sorted(self._tasks.items())
            )  # ensure tasks are sorted by index
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

    def get_description(self, task_id: TaskID) -> str:
        """Get the description of a task."""
        with self._lock:
            return self._tasks[task_id].description if task_id in self._tasks else ""
