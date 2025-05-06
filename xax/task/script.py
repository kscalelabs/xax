"""Composes various mixins into a single script interface."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import jax

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
)
from xax.task.mixins.profiler import ProfilerConfig, ProfilerMixin


@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class ScriptConfig(
    CPUStatsConfig,
    GPUStatsConfig,
    ProcessConfig,
    LoggerConfig,
    ArtifactsConfig,
    RunnableConfig,
    ProfilerConfig,
    BaseConfig,
):
    pass


ConfigT = TypeVar("ConfigT", bound=ScriptConfig)


class Script(
    CPUStatsMixin[ConfigT],
    GPUStatsMixin[ConfigT],
    ProcessMixin[ConfigT],
    LoggerMixin[ConfigT],
    ArtifactsMixin[ConfigT],
    RunnableMixin[ConfigT],
    ProfilerMixin[ConfigT],
    BaseTask[ConfigT],
    Generic[ConfigT],
):
    pass
