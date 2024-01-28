"""Defines a single interface for all the mixins."""

from mlfab.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin
from mlfab.task.mixins.checkpointing import CheckpointingConfig, CheckpointingMixin
from mlfab.task.mixins.compile import CompileConfig, CompileMixin
from mlfab.task.mixins.cpu_stats import CPUStatsConfig, CPUStatsMixin
from mlfab.task.mixins.data_loader import DataLoadersConfig, DataLoadersMixin
from mlfab.task.mixins.device import DeviceConfig, DeviceMixin
from mlfab.task.mixins.gpu_stats import GPUStatsConfig, GPUStatsMixin
from mlfab.task.mixins.logger import LoggerConfig, LoggerMixin
from mlfab.task.mixins.mixed_precision import MixedPrecisionConfig, MixedPrecisionMixin
from mlfab.task.mixins.optimizer import OptimizerConfig, OptimizerMixin
from mlfab.task.mixins.process import ProcessConfig, ProcessMixin
from mlfab.task.mixins.profiler import ProfilerConfig, ProfilerMixin
from mlfab.task.mixins.runnable import RunnableConfig, RunnableMixin
from mlfab.task.mixins.step_wrapper import StepContextConfig, StepContextMixin
from mlfab.task.mixins.train import TrainConfig, TrainMixin
