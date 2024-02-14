"""Defines the base launcher class."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from xax.task.base import RawConfigType

if TYPE_CHECKING:
    from xax.task.mixins.runnable import Config, RunnableMixin


class BaseLauncher(ABC):
    """Defines the base launcher class."""

    @abstractmethod
    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        """Launches the training process.

        Args:
            task: The task class to train
            cfgs: The raw configuration to use for training
            use_cli: Whether to include CLI arguments in the configuration
        """
