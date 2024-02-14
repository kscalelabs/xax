"""Composes various mixins into a single script interface."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from xax.task.base import BaseConfig, BaseTask
from xax.task.mixins import (
    ArtifactsConfig,
    ArtifactsMixin,
    CPUStatsConfig,
    CPUStatsMixin,
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
)


@dataclass
class ScriptConfig(
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


ConfigT = TypeVar("ConfigT", bound=ScriptConfig)


class Script(
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
