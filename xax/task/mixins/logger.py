"""Defines a mixin for incorporating some logging functionality."""

import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Callable, Generic, Self, TypeVar

from PIL.Image import Image as PILImage

from xax.core.conf import Device as BaseDeviceConfig, field
from xax.core.state import State
from xax.task.base import BaseConfig, BaseTask
from xax.task.logger import Logger, LoggerImpl, Number
from xax.task.loggers.json import JsonLogger
from xax.task.loggers.state import StateLogger
from xax.task.loggers.stdout import StdoutLogger
from xax.task.loggers.tensorboard import TensorboardLogger
from xax.task.mixins.artifacts import ArtifactsMixin
from xax.utils.text import is_interactive_session


@dataclass
class LoggerConfig(BaseConfig):
    device: BaseDeviceConfig = field(BaseDeviceConfig(), help="Device configuration")


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

    def log_scalar(self, key: str, value: Callable[[], Number] | Number, *, namespace: str | None = None) -> None:
        self.logger.log_scalar(key, value, namespace=namespace)

    def log_string(self, key: str, value: Callable[[], str] | str, *, namespace: str | None = None) -> None:
        self.logger.log_string(key, value, namespace=namespace)

    def log_image(
        self,
        key: str,
        value: Callable[[], PILImage] | PILImage,
        *,
        namespace: str | None = None,
    ) -> None:
        self.logger.log_image(
            key,
            value,
            namespace=namespace,
        )

    def log_images(
        self,
        key: str,
        value: Callable[[], list[PILImage]] | list[PILImage],
        *,
        namespace: str | None = None,
        max_images: int | None = None,
        sep: int = 0,
    ) -> None:
        self.logger.log_images(
            key,
            value,
            namespace=namespace,
            max_images=max_images,
            sep=sep,
        )

    def __enter__(self) -> Self:
        self.logger.__enter__()
        return self

    def __exit__(self, t: type[BaseException] | None, e: BaseException | None, tr: TracebackType | None) -> None:
        self.logger.__exit__(t, e, tr)
        return super().__exit__(t, e, tr)
