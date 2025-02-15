"""Defines a mixin for incorporating some logging functionality."""

import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Generic, Self, TypeVar

import jax

from xax.core.state import State
from xax.task.base import BaseConfig, BaseTask
from xax.task.logger import Logger, LoggerImpl
from xax.task.loggers.json import JsonLogger
from xax.task.loggers.state import StateLogger
from xax.task.loggers.stdout import StdoutLogger
from xax.task.loggers.tensorboard import TensorboardLogger
from xax.task.mixins.artifacts import ArtifactsMixin
from xax.utils.text import is_interactive_session


@jax.tree_util.register_dataclass
@dataclass
class LoggerConfig(BaseConfig):
    pass


Config = TypeVar("Config", bound=LoggerConfig)


def get_env_var(name: str, default: bool) -> bool:
    if name not in os.environ:
        return default
    return os.environ[name].strip() == "1"


class LoggerMixin(BaseTask[Config], Generic[Config]):
    logger: Logger

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.logger = Logger()

    def log_directory(self) -> Path | None:
        return None

    def add_logger(self, *logger: LoggerImpl) -> None:
        self.logger.add_logger(*logger)

    def set_loggers(self) -> None:
        self.add_logger(StdoutLogger() if is_interactive_session() else JsonLogger())
        if isinstance(self, ArtifactsMixin):
            self.add_logger(
                StateLogger(self.exp_dir),
                TensorboardLogger(self.exp_dir),
            )

    def write_logs(self, state: State) -> None:
        self.logger.write(state)

    def __enter__(self) -> Self:
        self.logger.__enter__()
        return self

    def __exit__(self, t: type[BaseException] | None, e: BaseException | None, tr: TracebackType | None) -> None:
        self.logger.__exit__(t, e, tr)
        return super().__exit__(t, e, tr)
