"""Defines a single interface for all the mixins."""

from xax.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin
from xax.task.mixins.cpu_stats import CPUStatsConfig, CPUStatsMixin
from xax.task.mixins.data_loader import DataloadersConfig, DataloadersMixin
from xax.task.mixins.gpu_stats import GPUStatsConfig, GPUStatsMixin
from xax.task.mixins.logger import LoggerConfig, LoggerMixin
from xax.task.mixins.process import ProcessConfig, ProcessMixin
from xax.task.mixins.runnable import RunnableConfig, RunnableMixin
from xax.task.mixins.step_wrapper import StepContextConfig, StepContextMixin
from xax.task.mixins.train import Batch, Model, Output, TrainConfig, TrainMixin
