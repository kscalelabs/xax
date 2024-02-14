"""Defines a mixin to wrap some steps in a context manager."""

from dataclasses import dataclass
from types import TracebackType
from typing import ContextManager, Literal, TypeVar

from xax.task.base import BaseConfig, BaseTask

StepType = Literal[
    "backward",
    "change_mode",
    "clip_grads",
    "create_optimizers",
    "forward",
    "get_dataloader",
    "get_dataset",
    "get_prefetcher",
    "get_model",
    "get_optimizer",
    "get_initial_opt_state",
    "get_update_fn",
    "load_checkpoint",
    "log_losses",
    "model_to_device",
    "on_step_end",
    "on_step_start",
    "save_checkpoint",
    "step",
    "update_state",
    "write_logs",
    "zero_grads",
]


class StepContext(ContextManager):
    """Context manager to get the current step type."""

    CURRENT_STEP: StepType | None = None

    def __init__(self, step: StepType) -> None:
        self.step = step

    def __enter__(self) -> None:
        StepContext.CURRENT_STEP = self.step

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        StepContext.CURRENT_STEP = None


@dataclass
class StepContextConfig(BaseConfig):
    pass


Config = TypeVar("Config", bound=StepContextConfig)


class StepContextMixin(BaseTask[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def step_context(self, step: StepType) -> ContextManager:
        return StepContext(step)
