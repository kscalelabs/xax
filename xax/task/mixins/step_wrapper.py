"""Defines a mixin to wrap some steps in a context manager."""

import time
from dataclasses import dataclass
from types import TracebackType
from typing import Callable, ContextManager, TypeVar

import jax

from xax.task.base import BaseConfig, BaseTask


class StepContext(ContextManager):
    """Context manager to get the current step type."""

    CURRENT_STEP: str | None = None

    def __init__(
        self,
        step: str,
        on_context_start: Callable[[str], None],
        on_context_end: Callable[[str, float], None],
    ) -> None:
        self.step = step
        self.start_time = 0.0
        self.on_context_start = on_context_start
        self.on_context_end = on_context_end

    def __enter__(self) -> None:
        StepContext.CURRENT_STEP = self.step
        self.start_time = time.time()
        self.on_context_start(self.step)

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        StepContext.CURRENT_STEP = None
        self.on_context_end(self.step, time.time() - self.start_time)


@jax.tree_util.register_dataclass
@dataclass
class StepContextConfig(BaseConfig):
    pass


Config = TypeVar("Config", bound=StepContextConfig)


class StepContextMixin(BaseTask[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def step_context(self, step: str) -> ContextManager:
        return StepContext(step, self.on_context_start, self.on_context_stop)

    def on_context_start(self, step: str) -> None:
        pass

    def on_context_stop(self, step: str, elapsed_time: float) -> None:
        pass
