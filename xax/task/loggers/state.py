"""Defines a logger which logs the current training state."""

from pathlib import Path
from typing import Literal

from xax.task.logger import LoggerImpl, LogLine


class StateLogger(LoggerImpl):
    def __init__(
        self,
        run_directory: str | Path,
        flush_immediately: bool = False,
        open_mode: Literal["w", "a"] = "w",
        line_sep: str = "\n",
        remove_unicode_from_namespaces: bool = True,
    ) -> None:
        super().__init__(float("inf"))

        self.run_directory = Path(run_directory).expanduser().resolve()

        self.flush_immediately = flush_immediately
        self.open_mode = open_mode
        self.line_sep = line_sep
        self.remove_unicode_from_namespaces = remove_unicode_from_namespaces

    def log_file(self, name: str, contents: str) -> None:
        with open(self.run_directory / name, "w") as f:
            f.write(contents)

    def write(self, line: LogLine) -> None:
        pass
