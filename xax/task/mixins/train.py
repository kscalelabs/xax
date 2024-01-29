"""Defines a mixin for running the training loop."""

import contextlib
import logging
import signal
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, is_dataclass
from threading import Thread
from typing import Any, Generic, Iterator, Literal, Mapping, Sequence, TypeVar, cast, get_args

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from omegaconf import DictConfig

from xax.core.conf import field
from xax.core.state import Phase, State
from xax.nn.functions import recursive_chunk
from xax.nn.parallel import is_master
from xax.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin
from xax.task.mixins.data_loader import DataLoadersConfig, DataLoadersMixin
from xax.task.mixins.logger import LoggerConfig, LoggerMixin
from xax.task.mixins.runnable import RunnableConfig, RunnableMixin
from xax.task.mixins.step_wrapper import StepContextConfig, StepContextMixin
from xax.utils.experiments import (
    EpochDoneError,
    StateTimer,
    TrainingFinishedError,
    get_git_state,
    get_training_code,
)
from xax.utils.logging import LOG_STATUS
from xax.utils.text import highlight_exception_message, show_info

logger = logging.getLogger(__name__)

# Batch = TypeVar("Batch")
# Output = TypeVar("Output")
# Input = TypeVar("Input")

Batch = Any
Output = Any
Input = Any

Loss = Array | dict[str, Array] | list[Array] | list[dict[str, Array]]

StepKind = Literal["step", "epoch", "sample", "second"]

PRINT_FINISH_TIME_EVERY_N_SECONDS = 60 * 2


def cast_step_kind(s: str) -> StepKind:
    assert s in get_args(StepKind), f"`step_kind` must be one of {get_args(StepKind)}, not {s}"
    return cast(StepKind, s)


@dataclass
class TrainConfig(
    DataLoadersConfig,
    LoggerConfig,
    StepContextConfig,
    ArtifactsConfig,
    RunnableConfig,
):
    valid_every_n_steps: int | None = field(None, help="Number of training steps to run per validation step")
    valid_every_n_seconds: float | None = field(60.0 * 10.0, help="Run validation every N seconds")
    valid_first_n_seconds: float | None = field(60.0, help="Run first validation after N seconds")
    batches_per_step: int = field(1, help="Batches to accumulate per training step, to simulate larger batch sizes")
    batches_per_step_schedule: list[int] | None = field(
        None,
        help=(
            "A schedule for increasing the effective batch size. The first segment will have batch size "
            "`batches_per_step`, the second will have `2 * batches_per_step`, the third will have "
            "`3 * batches_per_step`, and so on."
        ),
    )
    batch_chunks_per_step_schedule: list[int] | None = field(
        None,
        help=(
            "A schedule for splitting batches into chunks. The batches in the first segment will have "
            "`batch_size / (N + 1)` elements, the second will have `batch_size / N` elements, until "
            "the last segment has `batch_size` elements."
        ),
    )
    batch_dim: int = field(0, help="The batch dimension, for splitting batches into chunks")
    max_steps: int | None = field(None, help="Maximum number of steps to run")
    step_kind: str = field("step", help=f"How to measure a step; one of [{', '.join(get_args(StepKind))}]")


Config = TypeVar("Config", bound=TrainConfig)


class TrainMixin(
    DataLoadersMixin[Config],
    LoggerMixin[Config],
    StepContextMixin[Config],
    ArtifactsMixin[Config],
    RunnableMixin[Config],
    Generic[Config],
):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.last_valid_time: float | None = None
        self.last_valid_step: int | None = None
        self.first_valid_step_flag = True

        # Timers for iterations.
        self.state_timers: dict[Phase, StateTimer] = {phase: StateTimer() for phase in get_args(Phase)}

        # This flag can be toggled to end training from anywhere in the task.
        self.__training_over_flag = False

        self.__last_printed_remaining_time = 0.0

        # The kind of step that was specified in the config.
        self.__step_kind = cast_step_kind(self.config.step_kind)

    def on_step_end(self, state: State, loss_dict: dict[str, Array]) -> None:
        super().on_step_end(state, loss_dict)
        state.elapsed_time_s = time.time() - state.start_time_s

    def on_epoch_start(self, state: State) -> None:
        super().on_epoch_start(state)
        state.num_epoch_steps = 0
        state.num_epoch_samples = 0

    def on_epoch_end(self, state: State) -> None:
        super().on_epoch_end(state)
        state.num_epochs += 1

    def log_train_step(self, batch: Batch, output: Output, state: State) -> None:
        """Override this function to do logging during the training phase.

        This function is called after the model forward pass and before the
        backward pass. It is called in the training phase.

        Args:
            batch: The batch from the dataloader.
            output: The model output.
            state: The current training state.
        """

    def log_valid_step(self, batch: Batch, output: Output, state: State) -> None:
        """Override this function to do logging during the validation phase.

        This function is called after the model forward pass. It is called in
        the validation phase.

        Args:
            batch: The batch from the dataloader.
            output: The model output.
            state: The current training state.
        """

    def log_test_step(self, batch: Batch, output: Output, state: State) -> None:
        """Override this function to do logging during the test phase.

        This function is called after the model forward pass. It is called in
        the validation phase.

        Args:
            batch: The batch from the dataloader.
            output: The model output.
            state: The current training state.
        """
        return self.log_valid_step(batch, output, state)

    def log_step(self, batch: Batch, output: Output, state: State) -> None:
        match state.phase:
            case "train":
                self.log_train_step(batch, output, state)
            case "valid":
                self.log_valid_step(batch, output, state)
            case "test":
                self.log_test_step(batch, output, state)
            case _:
                raise KeyError(f"Unknown phase: {state.phase}")

    def get_loss(self, batch: Batch, state: State) -> Loss:
        """Gets the loss for the current batch.

        By default, we assume the model's forward function takes the batch as
        input and returns the loss. We do some logging with the output, and
        return it as the loss. This function can be patched to do more complex
        operations instead.

        Args:
            batch: The current minibatch of samples.
            state: The current training state.

        Returns:
            The computed loss or losses, either a tensor or dictionary of
            tensors. The dictionary keys are used when logging the losses.
        """
        output = self(batch)
        self.log_step(batch, output, state)
        return output

    def get_size_of_batch(self, batch: Batch) -> int | None:
        """Gets the batch size for the current batch.

        Args:
            batch: The current minibatch of samples.

        Returns:
            The parsed batch size, or None if the batch size could not be
            determined.
        """
        if isinstance(batch, (np.ndarray, Array)):
            return batch.shape[0]
        if is_dataclass(batch):
            for v in batch.__dict__.values():
                if bsz := self.get_size_of_batch(v):
                    return bsz
        if isinstance(batch, Mapping):
            for v in batch.values():
                if bsz := self.get_size_of_batch(v):
                    return bsz
        if isinstance(batch, Sequence):
            for i in batch:
                if bsz := self.get_size_of_batch(i):
                    return bsz
        return None

    def set_training_over(self) -> None:
        self.__training_over_flag = True

    def maybe_log_termination_time(self, remaining_percent: float, state: State) -> None:
        if self.__last_printed_remaining_time + PRINT_FINISH_TIME_EVERY_N_SECONDS > state.elapsed_time_s:
            return
        self.__last_printed_remaining_time = state.elapsed_time_s
        remaining_seconds = remaining_percent * state.elapsed_time_s / (1 - remaining_percent)
        termination_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining_seconds))
        logger.info("Estimated finish time: %s", termination_time)

    def is_training_over(self, state: State) -> bool:
        if self.__training_over_flag:
            return True
        remaining_percent = self.get_remaining_percent(state)
        if remaining_percent is None:
            return False
        self.log_scalar("percent", remaining_percent, namespace="⏰ remaining")
        self.maybe_log_termination_time(remaining_percent, state)
        return remaining_percent <= 0.0

    def get_step(self, state: State) -> int:
        match self.__step_kind:
            case "step":
                return state.num_steps
            case "epoch":
                return state.num_epochs
            case "sample":
                return state.num_samples
            case "second":
                return int(state.elapsed_time_s)
            case _:
                raise ValueError(f"Invalid step kind {self.__step_kind}")

    def get_remaining_percent(self, state: State) -> float | None:
        if self.config.max_steps is None:
            return None
        return (self.config.max_steps - self.get_step(state)) / self.config.max_steps

    def get_single_loss(self, loss: Array | dict[str, Array]) -> tuple[Array, list[str]]:
        if isinstance(loss, Array):
            return loss.sum() / loss.shape[self.config.batch_dim], ["loss"]
        if isinstance(loss, dict):
            keys, values = (list(i) for i in zip(*sorted(loss.items())))
            losses = [v.sum() / v.shape[0] if v.ndim > 0 else v for v in values]
            single_loss = jnp.stack(losses, axis=0)
            return single_loss, keys
        raise NotImplementedError(f"Loss should be a scalar, dictionary, or list, not {type(loss)}")

    def get_single_losses(self, losses: Loss) -> list[tuple[Array, list[str]]]:
        if isinstance(losses, (Array, dict)):
            return [self.get_single_loss(losses)]
        assert isinstance(losses, list), f"Loss should be a scalar, dictionary, or list, not {type(losses)}"
        single_losses = [self.get_single_loss(loss) for loss in losses]
        return single_losses

    def log_loss_dict(self, loss: Mapping[str, int | float | Array], state: State) -> None:
        for k, v in loss.items():
            self.log_scalar(k, v, namespace="loss")
        timer = self.state_timers[state.phase]
        timer.step(state)
        for ns, d in timer.log_dict().items():
            for k, v in d.items():
                self.log_scalar(k, v, namespace=ns)

    def is_valid_step(self, state: State) -> bool:
        if self.last_valid_time is None or self.last_valid_step is None:
            self.last_valid_time = state.elapsed_time_s
            self.last_valid_step = state.num_steps
            return True

        # Step-based validation.
        valid_every_n_steps = self.config.valid_every_n_steps
        if valid_every_n_steps is not None and state.num_steps > valid_every_n_steps + self.last_valid_step:
            self.last_valid_step = state.num_steps
            return True

        # Time-based validation.
        valid_every_n_seconds = self.config.valid_every_n_seconds
        if valid_every_n_seconds is not None and state.elapsed_time_s - self.last_valid_time >= valid_every_n_seconds:
            self.last_valid_time = state.elapsed_time_s
            return True

        # Time-based validation for first validation step.
        if self.first_valid_step_flag:
            valid_first_n_seconds = self.config.valid_first_n_seconds
            if valid_first_n_seconds is not None and state.elapsed_time_s >= valid_first_n_seconds:
                self.last_valid_time = state.elapsed_time_s
                self.first_valid_step_flag = False
                return True

        return False

    def log_state(self) -> None:
        logger.log(LOG_STATUS, self.task_path)
        logger.log(LOG_STATUS, self.task_name)
        self.logger.log_git_state(get_git_state(self))
        self.logger.log_training_code(get_training_code(self))
        self.logger.log_config(cast(DictConfig, self.config))

    def run(self) -> None:
        self.run_training_loop()

    def run_training_loop(self) -> None:
        """Runs the training loop.

        Args:
            model: The current model
            task: The current task
            optimizer: The current optimizer
            lr_scheduler: The current learning rate scheduler

        Raises:
            ValueError: If the task is not a supervised learning task
        """
        self.set_loggers()

        breakpoint()

        with self.step_context("model_to_device"):
            self.device.module_to(self)

        with self.step_context("create_optimizers"):
            self.set_optimizers()

        if is_master():
            Thread(target=self.log_state, daemon=True).start()

        with self.step_context("load_checkpoint"):
            state = self.load_initial_state()

        # Gets the datasets.
        with self.step_context("get_dataset"):
            train_ds = self.get_dataset("train")
            valid_ds = self.get_dataset("valid")

        # Gets the dataloaders.
        with self.step_context("get_dataloader"):
            train_dl = self.get_dataloader(train_ds, "train")
            valid_dl = self.get_dataloader(valid_ds, "valid")

        # Gets the prefetchers.
        with self.step_context("get_prefetcher"):
            train_pf = self.device.get_prefetcher(train_dl)
            valid_pf = self.device.get_prefetcher(valid_dl)
            valid_pf_iter = iter(valid_pf.infinite())

        self.on_training_start(state)

        def on_exit() -> None:
            self.save_checkpoint(state)

        # Handle user-defined interrupts during the training loop.
        self.add_signal_handler(on_exit, signal.SIGUSR1, signal.SIGTERM)

        try:
            with contextlib.ExitStack() as ctx:
                ctx.enter_context(self)

                if (profile := self.get_profile()) is not None:
                    ctx.enter_context(profile)

                while True:
                    with self.step_context("on_epoch_start"):
                        self.on_epoch_start(state)

                    def batch_splitter() -> Iterator[Batch]:
                        for batch in train_pf:
                            num_chunks = self.get_batch_chunks(state)
                            yield from recursive_chunk(batch, num_chunks, dim=self.config.batch_dim)

                    train_pf_iter: Iterator = batch_splitter()

                    def batch_iterator() -> Iterator[Batch]:
                        try:
                            yield next(train_pf_iter)
                        except StopIteration:
                            raise EpochDoneError

                        for _ in range(self.get_batches_per_step(state) - 1):
                            try:
                                yield next(train_pf_iter)
                            except StopIteration:
                                pass

                    while True:
                        if self.is_training_over(state):
                            raise TrainingFinishedError

                        if self.is_valid_step(state):
                            self.val_step(next(valid_pf_iter), state)

                        with self.step_context("on_step_start"):
                            self.on_step_start(state)

                        try:
                            loss_dict = self.train_step(batch_iterator(), state)

                        except EpochDoneError:
                            break

                        if self.should_checkpoint(state):
                            with self.step_context("save_checkpoint"):
                                self.save_checkpoint(state)

                        if profile is not None:
                            profile.step()

                        with self.step_context("on_step_end"):
                            self.on_step_end(state, loss_dict)

                    with self.step_context("on_epoch_end"):
                        self.on_epoch_end(state)

        except TrainingFinishedError:
            with self.step_context("save_checkpoint"):
                self.save_checkpoint(state)
            if is_master():
                show_info(
                    f"Finished training after {state.num_epochs} epochs, "
                    f"{state.num_steps} steps, {state.num_samples} samples",
                    important=True,
                )

        except BaseException:
            exception_tb = textwrap.indent(highlight_exception_message(traceback.format_exc()), "  ")
            sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
            sys.stdout.flush()

        finally:
            self.on_training_end(state)
