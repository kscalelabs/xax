"""Composes various mixins into a single script interface."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from xax.task.base import BaseConfig, BaseTask
from xax.task.mixins import (
    ArtifactsConfig,
    ArtifactsMixin,
    CPUStatsConfig,
    CPUStatsMixin,
    DeviceConfig,
    DeviceMixin,
    GPUStatsConfig,
    GPUStatsMixin,
    LoggerConfig,
    LoggerMixin,
    MixedPrecisionConfig,
    MixedPrecisionMixin,
    ProcessConfig,
    ProcessMixin,
    ProfilerConfig,
    ProfilerMixin,
    RunnableConfig,
    RunnableMixin,
    StepContextConfig,
    StepContextMixin,
)


@dataclass
class ScriptConfig(
    MixedPrecisionConfig,
    CPUStatsConfig,
    DeviceConfig,
    GPUStatsConfig,
    ProcessConfig,
    ProfilerConfig,
    LoggerConfig,
    StepContextConfig,
    ArtifactsConfig,
    RunnableConfig,
    BaseConfig,
):
    pass


ConfigT = TypeVar("ConfigT", bound=ScriptConfig)


class Script(
    MixedPrecisionMixin[ConfigT],
    CPUStatsMixin[ConfigT],
    DeviceMixin[ConfigT],
    GPUStatsMixin[ConfigT],
    ProcessMixin[ConfigT],
    ProfilerMixin[ConfigT],
    LoggerMixin[ConfigT],
    StepContextMixin[ConfigT],
    ArtifactsMixin[ConfigT],
    RunnableMixin[ConfigT],
    BaseTask[ConfigT],
    Generic[ConfigT],
):
    pass
