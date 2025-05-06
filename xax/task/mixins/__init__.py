"""Task mixins that add capabilities to a task."""

from xax.task.mixins.artifacts import (
    ArtifactConfig,
    ArtifactsMixin,
)
from xax.task.mixins.checkpointing import (
    CheckpointConfig,
    CheckpointingMixin,
)
from xax.task.mixins.compile import (
    CompileConfig,
    CompileMixin,
)
from xax.task.mixins.cpu_stats import (
    CPUStatsConfig,
    CPUStatsMixin,
)
from xax.task.mixins.data_loader import (
    DataLoaderConfig,
    DataLoaderMixin,
)
from xax.task.mixins.gpu_stats import (
    GPUStatsConfig,
    GPUStatsMixin,
)
from xax.task.mixins.logger import (
    LoggerConfig,
    LoggerMixin,
)
from xax.task.mixins.mixed_precision import (
    MixedPrecisionConfig,
    MixedPrecisionMixin,
)
from xax.task.mixins.process import (
    ProcessConfig,
    ProcessMixin,
)
from xax.task.mixins.runnable import (
    RunnableConfig,
    RunnableMixin,
)
from xax.task.mixins.step_wrapper import (
    StepWrapperConfig,
    StepWrapperMixin,
)
from xax.task.mixins.train import (
    TrainConfig,
    TrainMixin,
)

__all__ = [
    "ArtifactConfig",
    "ArtifactsMixin",
    "CheckpointConfig",
    "CheckpointingMixin",
    "CompileConfig",
    "CompileMixin",
    "CPUStatsConfig",
    "CPUStatsMixin",
    "DataLoaderConfig",
    "DataLoaderMixin",
    "GPUStatsConfig",
    "GPUStatsMixin",
    "LoggerConfig",
    "LoggerMixin",
    "MixedPrecisionConfig",
    "MixedPrecisionMixin",
    "ProcessConfig",
    "ProcessMixin",
    "RunnableConfig",
    "RunnableMixin",
    "StepWrapperConfig",
    "StepWrapperMixin",
    "TrainConfig",
    "TrainMixin",
]
