"""Composes the base task with all the mixins into a single task interface."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import jax

from xax.task.base import BaseConfig, BaseTask
from xax.task.mixins import (
    ArtifactsConfig,
    ArtifactsMixin,
    CheckpointingConfig,
    CheckpointingMixin,
    CompileConfig,
    CompileMixin,
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


@jax.tree_util.register_dataclass
@dataclass
class Config(
    TrainConfig,
    CheckpointingConfig,
    CompileConfig,
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
    CompileMixin[ConfigT],
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
