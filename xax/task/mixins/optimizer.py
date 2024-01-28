"""Defines a mixin which supports an optimizer and learning rate scheduler."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Generic, TypeVar

from mlfab.core.conf import field
from mlfab.nn.lr_schedulers import BaseLRScheduler, ConstantLRScheduler, SchedulerAdapter
from mlfab.task.base import BaseConfig, BaseTask
from torch.optim.optimizer import Optimizer

OptType = (
    Optimizer
    | tuple[Optimizer, BaseLRScheduler]
    | Collection[Optimizer]
    | Collection[tuple[Optimizer, BaseLRScheduler]]
)


@dataclass
class OptimizerConfig(BaseConfig):
    set_grads_to_none: bool = field(True, help="If set, zero gradients by setting them to None")


Config = TypeVar("Config", bound=OptimizerConfig)


class OptimizerMixin(BaseTask[Config], Generic[Config], ABC):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._optimizers: list[SchedulerAdapter] | None = None

    @abstractmethod
    def build_optimizer(self) -> OptType:
        """Gets the optimizer or optimizers for the current model.

        If the return type is a single optimizer, then a constant learning rate
        will be used.

        Returns:
            The optimizer, or paired optimizer and learning rate scheduler, or
            a list of optimizers, or a list of optimizers paired with learning
            rate schedulers.
        """

    @property
    def optimizers(self) -> list[SchedulerAdapter]:
        if self._optimizers is None:
            raise AttributeError("Cannot access optimizers yet; must call `set_optimizers` first!")
        return self._optimizers

    def zero_optimizers(self) -> None:
        for optim_i in self.optimizers:
            optim_i.optimizer.zero_grad(set_to_none=self.config.set_grads_to_none)

    def set_optimizers(self) -> None:
        if self._optimizers is not None:
            raise RuntimeError("Optimizers have already been set!")
        optimizers = self.build_optimizer()

        if isinstance(optimizers, Optimizer):
            self._optimizers = [ConstantLRScheduler().get(optimizers)]

        elif isinstance(optimizers, tuple):
            if (
                len(optimizers) == 2
                and isinstance(optimizers[0], Optimizer)
                and isinstance(optimizers[1], BaseLRScheduler)
            ):
                opt, sched = optimizers
                self._optimizers = [sched.get(opt)]
            else:
                raise ValueError(
                    "Got invalid tuple from `build_optimizer`. If returning a tuple, it must be of the form "
                    "`(optimizer, learning_rate_scheduler)`, where the optimizer is a PyTorch optimizer and the "
                    "learning rate scheduler is a `BaseLRScheduler` instance."
                )

        elif isinstance(optimizers, Collection):
            if all(
                isinstance(opt, tuple)
                and len(opt) == 2
                and isinstance(opt[0], Optimizer)
                and isinstance(opt[1], BaseLRScheduler)
                for opt in optimizers
            ):
                self._optimizers = [sched.get(opt) for opt, sched in optimizers]  # type: ignore[misc]
            else:
                raise ValueError(
                    "Got invalid collection from `build_optimizer`. If returning a collection, it must be a "
                    "collection of tuples of the form `(optimizer, learning_rate_scheduler)`, where the optimizer "
                    "is a PyTorch optimizer and the learning rate scheduler is a `BaseLRScheduler` instance."
                )

        else:
            raise ValueError(
                "Unexpected return value from `build_optimizer`. This function should return a PyTorch optimizer, "
                "or a tuple of a PyTorch optimizer and learning rate scheduler, or a list of tuples of PyTorch "
                "optimizers and learning rate schedulers."
            )

    def load_task_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False) -> None:
        if self._optimizers is None:
            return super().load_task_state_dict(state_dict, strict, assign)
        optimizer_states = state_dict.pop("optimizers", [])
        if len(self._optimizers) != len(optimizer_states):
            raise ValueError(
                f"Invalid state dict; module has {len(self._optimizers)} optimizer(s) "
                f"but state dict has {len(optimizer_states)} optimizer state(s)"
            )
        for optimizer_state, optimizer in zip(optimizer_states, self._optimizers):
            optimizer.load_state_dict(optimizer_state)
        return super().load_task_state_dict(state_dict, strict, assign)

    def task_state_dict(self) -> dict:
        state_dict = super().task_state_dict()
        if self._optimizers is not None:
            state_dict.update({"optimizers": [opt.state_dict() for opt in self._optimizers]})
        return state_dict
