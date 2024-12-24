"""Composes the base task with all the mixins into a single task interface."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from xax.task.base import BaseConfig, BaseTask
from xax.task.mixins import (
    ArtifactsConfig,
    ArtifactsMixin,
    CheckpointingConfig,
    CheckpointingMixin,
    CPUStatsConfig,
    CPUStatsMixin,
    DataloadersConfig,
    DataloadersMixin,
    GPUStatsConfig,
    GPUStatsMixin,
    LoggerConfig,
    LoggerMixin,
    ProcessConfig,
    ProcessMixin,
    RunnableConfig,
    RunnableMixin,
    StepContextConfig,
    StepContextMixin,
    TrainConfig,
    TrainMixin,
)


@dataclass(kw_only=True)
class Config(
    TrainConfig,
    CheckpointingConfig,
    DataloadersConfig,
    CPUStatsConfig,
    GPUStatsConfig,
    ProcessConfig,
    LoggerConfig,
    StepContextConfig,
    ArtifactsConfig,
    RunnableConfig,
    BaseConfig,
):
    pass


ConfigT = TypeVar("ConfigT", bound=Config)


class Task(
    TrainMixin[ConfigT],
    CheckpointingMixin[ConfigT],
    DataloadersMixin[ConfigT],
    CPUStatsMixin[ConfigT],
    GPUStatsMixin[ConfigT],
    ProcessMixin[ConfigT],
    LoggerMixin[ConfigT],
    StepContextMixin[ConfigT],
    ArtifactsMixin[ConfigT],
    RunnableMixin[ConfigT],
    BaseTask[ConfigT],
    Generic[ConfigT],
):
    pass
