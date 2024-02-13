"""Defines a mixin which supports an optimizer and learning rate scheduler."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Generic, TypeVar

import optax

from xax.task.base import BaseConfig, BaseTask

OptType = optax.GradientTransformation | Collection[optax.GradientTransformation]


@dataclass
class OptimizerConfig(BaseConfig):
    pass


Config = TypeVar("Config", bound=OptimizerConfig)


class OptimizerMixin(BaseTask[Config], Generic[Config], ABC):
    optimizers: list[optax.GradientTransformation]

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Builds the optimizer or optimizers.
        optimizers = self.get_optimizer()
        if isinstance(optimizers, list):
            if not all(isinstance(o, optax.GradientTransformation) for o in optimizers):
                raise ValueError("All optimizers must be of type optax.GradientTransformation")
            self.optimizers = optimizers
        elif isinstance(optimizers, optax.GradientTransformation):
            self.optimizers = [optimizers]
        else:
            raise ValueError("Optimizer must be of type optax.GradientTransformation")

    @abstractmethod
    def get_optimizer(self) -> OptType:
        """Gets the optimizer or optimizers for the current model.

        If the return type is a single optimizer, then a constant learning rate
        will be used.

        Returns:
            The optimizer, or paired optimizer and learning rate scheduler, or
            a list of optimizers, or a list of optimizers paired with learning
            rate schedulers.
        """
