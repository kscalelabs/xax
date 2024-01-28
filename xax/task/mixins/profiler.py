"""Defines a task mixin for profiling PyTorch models."""

import contextlib
import datetime
import logging
import time
from dataclasses import dataclass
from typing import Any, ContextManager, Generic, Iterator, TypeVar

import torch
from mlfab.core.conf import field
from mlfab.nn.device.gpu import gpu_device
from mlfab.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin
from mlfab.task.mixins.logger import LoggerConfig, LoggerMixin
from mlfab.task.mixins.step_wrapper import StepContextConfig, StepContextMixin, StepType

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class ProfilerOptions:
    enabled: bool = field(False, help="If profiling should be enabled")
    record_shapes: bool = field(False, help="If set, record tensor shapes")
    profile_memory: bool = field(False, help="If set, profile PyTorch memory")
    with_stack: bool = field(False, help="Record source information (file and line number) for ops")
    with_flops: bool = field(False, help="Use formula to estimate the FLOPs of specific operations")
    with_modules: bool = field(False, help="Record module hierarchy (including function names)")
    wait_steps: int = field(10, help="Number of initial waiting steps")
    warmup_steps: int = field(10, help="Number of profiler warmup steps")
    active_steps: int = field(10, help="Number of profiler active steps")
    repeat_steps: int = field(1, help="Number of profiler repetitions")
    skip_first_steps: int = field(10, help="Number of profiler steps to skip at first")
    table_size: int = field(10, help="Number of profiling ops to print")


STEPS_TO_TIME: set[StepType] = {
    "backward",
    "clip_grads",
    "forward",
    "get_single_loss",
    "log_losses",
    "on_step_end",
    "on_step_start",
    "step",
    "write_logs",
    "zero_grads",
}


STEPS_TO_WARN_IF_LONG: set[StepType] = {
    "create_optimizers",
    "get_dataloader",
    "get_dataset",
    "get_prefetcher",
    "load_checkpoint",
    "model_to_device",
    "save_checkpoint",
}


@dataclass
class ProfilerConfig(LoggerConfig, StepContextConfig, ArtifactsConfig):
    profiler: ProfilerOptions = field(ProfilerOptions(), help="Profiler configuration")


Config = TypeVar("Config", bound=ProfilerConfig)


class ProfilerMixin(
    LoggerMixin[Config],
    StepContextMixin[Config],
    ArtifactsMixin[Config],
    Generic[Config],
):
    """Defines a task mixin for enabling the PyTorch profiler."""

    def warn_if_step_too_long(self, step: StepType, seconds: float) -> bool:
        return seconds > 5.0

    def step_context(self, step: StepType) -> ContextManager:
        ctx = super().step_context(step)

        if step in STEPS_TO_TIME:

            @contextlib.contextmanager
            def wrapped_timer_ctx() -> Iterator[Any]:
                start_time = time.time()

                if self.config.profiler.enabled:
                    with ctx as a, torch.profiler.record_function(step) as b:
                        yield a, b
                else:
                    with ctx as a:
                        yield a

                step_time = time.time() - start_time
                self.log_scalar(step, step_time, namespace="🔧 dt")

            return wrapped_timer_ctx()

        if step in STEPS_TO_WARN_IF_LONG:

            @contextlib.contextmanager
            def wrapped_warn_if_long_ctx() -> Iterator[Any]:
                start_time = time.time()

                if self.config.profiler.enabled:
                    with ctx as a, torch.profiler.record_function(step) as b:
                        yield a, b
                else:
                    with ctx as a:
                        yield a

                step_time = time.time() - start_time
                if self.warn_if_step_too_long(step, step_time):
                    logger.warning("Step %s took %.2f seconds", step, step_time)

            return wrapped_warn_if_long_ctx()

        return ctx

    def on_profiler_trace_ready(self, prof: torch.profiler.profile) -> None:
        key_averages = prof.key_averages()

        # Prints a table with informative statistics.
        keys = ["self_cpu_time_total", "cpu_time_total", "cpu_memory_usage"]
        if gpu_device.has_device():
            keys += ["self_cuda_time_total", "cuda_time_total", "cuda_memory_usage"]
        for key in keys:
            table = key_averages.table(
                sort_by=key,
                row_limit=self.config.profiler.table_size,
                top_level_events_only=False,
            )
            logger.info("%s:\n%s", key, table)

        # Saves a stack trace that is viewable in Chrome, in chrome://tracing/
        profile_dir = self.exp_dir / "profile"
        profile_dir.mkdir(exist_ok=True, parents=True)
        date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        prof.export_chrome_trace(str(profile_dir / f"trace.step_{prof.step_num}.{date_str}.json"))

    def get_profile(self) -> torch.profiler.profile | None:
        if not self.config.profiler.enabled:
            return None

        if gpu_device.has_device():
            profiler_activities = [
                torch.autograd.ProfilerActivity.CPU,
                torch.autograd.ProfilerActivity.CUDA,
            ]
        else:
            profiler_activities = [
                torch.autograd.ProfilerActivity.CPU,
            ]

        return torch.profiler.profile(
            activities=profiler_activities,
            record_shapes=self.config.profiler.record_shapes,
            profile_memory=self.config.profiler.profile_memory,
            with_stack=self.config.profiler.with_stack,
            with_flops=self.config.profiler.with_flops,
            with_modules=self.config.profiler.with_modules,
            schedule=torch.profiler.schedule(
                wait=self.config.profiler.wait_steps,
                warmup=self.config.profiler.warmup_steps,
                active=self.config.profiler.active_steps,
                repeat=self.config.profiler.repeat_steps,
                skip_first=self.config.profiler.skip_first_steps,
            ),
            on_trace_ready=self.on_profiler_trace_ready,
        )
