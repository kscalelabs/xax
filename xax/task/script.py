"""Composes various mixins into a single script interface."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from mlfab.task.base import BaseConfig, BaseTask
from mlfab.task.mixins import (
    ArtifactsConfig,
    ArtifactsMixin,
    CompileConfig,
    CompileMixin,
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
    CompileConfig,
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
    CompileMixin[ConfigT],
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
