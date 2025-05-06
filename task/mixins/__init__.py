"""Mixins for XAX tasks."""

from xax.task.mixins.train import TrainConfig, TrainMixin
from xax.task.mixins.mixed_precision import MixedPrecisionConfig, MixedPrecisionMixin

__all__ = [
    "TrainConfig",
    "TrainMixin",
    "MixedPrecisionConfig",
    "MixedPrecisionMixin",
] 