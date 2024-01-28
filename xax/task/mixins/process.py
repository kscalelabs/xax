"""Defines a base trainer mixin for handling subprocess monitoring jobs."""

import logging
import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Generic, TypeVar

from mlfab.core.conf import field
from mlfab.core.state import State
from mlfab.task.base import BaseConfig, BaseTask

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class ProcessConfig(BaseConfig):
    multiprocessing_context: str | None = field(None, help="The multiprocessing context to use")


Config = TypeVar("Config", bound=ProcessConfig)


class ProcessMixin(BaseTask[Config], Generic[Config]):
    """Defines a base trainer mixin for handling monitoring processes."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._mp_ctx = mp.get_context(config.multiprocessing_context)
        self._mp_manager = self._mp_ctx.Manager()

    @property
    def multiprocessing_context(self) -> BaseContext:
        return self._mp_ctx

    def on_training_start(self, state: State) -> None:
        super().on_training_start(state)

        self._mp_manager = mp.Manager()

    def on_training_end(self, state: State) -> None:
        super().on_training_end(state)

        self._mp_manager.shutdown()
        self._mp_manager.join()
