"""Defines a mixin for running the training loop."""

import bdb
import contextlib
import functools
import itertools
import logging
import signal
import sys
import textwrap
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from threading import Thread
from typing import (
    Any,
    Generator,
    Generic,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    cast,
    get_args,
    overload,
)

import equinox as eqx
import jax
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree
from omegaconf import DictConfig

from xax.core.conf import field
from xax.core.state import Phase, State
from xax.nn.functions import set_random_seed
from xax.nn.parallel import is_master
from xax.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin
from xax.task.mixins.checkpointing import CheckpointingConfig, CheckpointingMixin
from xax.task.mixins.data_loader import DataloadersConfig, DataloadersMixin
from xax.task.mixins.logger import LoggerConfig, LoggerMixin
from xax.task.mixins.runnable import RunnableConfig, RunnableMixin
from xax.task.mixins.step_wrapper import StepContextConfig, StepContextMixin
from xax.utils.experiments import (
    StateTimer,
    TrainingFinishedError,
    diff_configs,
    get_diff_string,
    get_git_state,
    get_training_code,
)
from xax.utils.logging import LOG_STATUS
from xax.utils.text import highlight_exception_message, show_info

logger = logging.getLogger(__name__)

# Batch = TypeVar("Batch")
# Output = TypeVar("Output")

Batch = Any
Output = Any

StepKind = Literal["step", "sample", "second"]

PRINT_FINISH_TIME_EVERY_N_SECONDS = 60 * 2


def cast_step_kind(s: str) -> StepKind:
    assert s in get_args(StepKind), f"`step_kind` must be one of {get_args(StepKind)}, not {s}"
    return cast(StepKind, s)


@functools.lru_cache(maxsize=None)
def batch_chunks_schedule(schedule: list[int] | None) -> list[int] | None:
    if schedule is None:
        return None
    if any(s < 1 for s in schedule):
        raise ValueError("Batch chunk schedule must be positive")
    return list(itertools.accumulate([0] + schedule))


@functools.lru_cache(maxsize=None)
def batches_per_step_schedule(schedule: list[int] | None) -> list[int] | None:
    if schedule is None:
        return None
    if any(s < 1 for s in schedule):
        raise ValueError("Batch chunk schedule must be positive")
    return list(itertools.accumulate([0] + schedule))


class ValidStepTimer:
    def __init__(
        self,
        valid_every_n_steps: int | None = None,
        valid_first_n_steps: int = 0,
        valid_every_n_seconds: float | None = None,
        valid_first_n_seconds: float | None = None,
    ) -> None:
        super().__init__()

        self.valid_every_n_steps = valid_every_n_steps
        self.valid_first_n_steps = valid_first_n_steps
        self.valid_every_n_seconds = valid_every_n_seconds
        self.valid_first_n_seconds = valid_first_n_seconds
        self.first_valid_step_flag = True

        self.last_valid_time: float | None = None
        self.last_valid_step: int | None = None

    def is_valid_step(self, state: State) -> bool:
        if state.num_steps < self.valid_first_n_steps:
            return True

        if self.last_valid_time is None or self.last_valid_step is None:
            self.last_valid_time = state.elapsed_time_s
            self.last_valid_step = state.num_steps
            return False

        # Step-based validation.
        valid_every_n_steps = self.valid_every_n_steps
        if valid_every_n_steps is not None and state.num_steps > valid_every_n_steps + self.last_valid_step:
            self.last_valid_step = state.num_steps
            return True

        # Time-based validation.
        valid_every_n_seconds = self.valid_every_n_seconds
        if valid_every_n_seconds is not None and state.elapsed_time_s - self.last_valid_time >= valid_every_n_seconds:
            self.last_valid_time = state.elapsed_time_s
            return True

        # Time-based validation for first validation step.
        if self.first_valid_step_flag:
            valid_first_n_seconds = self.valid_first_n_seconds
            if valid_first_n_seconds is not None and state.elapsed_time_s >= valid_first_n_seconds:
                self.last_valid_time = state.elapsed_time_s
                self.first_valid_step_flag = False
                return True

        return False


@jax.tree_util.register_dataclass
@dataclass
class TrainConfig(
    CheckpointingConfig,
    DataloadersConfig,
    LoggerConfig,
    StepContextConfig,
    ArtifactsConfig,
    RunnableConfig,
):
    valid_every_n_steps: int | None = field(None, help="Number of training steps to run per validation step")
    valid_first_n_steps: int = field(0, help="Treat the first N steps as validation steps")
    valid_every_n_seconds: float | None = field(60.0 * 10.0, help="Run validation every N seconds")
    valid_first_n_seconds: float | None = field(60.0, help="Run first validation after N seconds")
    max_steps: int | None = field(None, help="Maximum number of steps to run")
    step_kind: str = field("step", help=f"How to measure a step; one of [{', '.join(get_args(StepKind))}]")
    random_seed: int = field(1337, help="Random seed for the task")


Config = TypeVar("Config", bound=TrainConfig)


class TrainMixin(
    CheckpointingMixin[Config],
    DataloadersMixin[Config],
    LoggerMixin[Config],
    StepContextMixin[Config],
    ArtifactsMixin[Config],
    RunnableMixin[Config],
    Generic[Config],
    ABC,
):
    valid_step_timer: ValidStepTimer
    state_timers: dict[Phase, StateTimer]

    _training_over_flag: bool
    _last_printed_remaining_time: float
    _step_kind: StepKind

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Sets the random seed whenever we instantiate a new train mixin.
        set_random_seed(self.config.random_seed)

        # Timer for validation steps.
        self.valid_step_timer = ValidStepTimer(
            valid_every_n_steps=config.valid_every_n_steps,
            valid_first_n_steps=config.valid_first_n_steps,
            valid_every_n_seconds=config.valid_every_n_seconds,
            valid_first_n_seconds=config.valid_first_n_seconds,
        )

        # Timers for iterations.
        self.state_timers = {phase: StateTimer() for phase in get_args(Phase)}

        # This flag can be toggled to end training from anywhere in the task.
        self._training_over_flag = False

        self._last_printed_remaining_time = 0.0

        # The kind of step that was specified in the config.
        self._step_kind = cast_step_kind(self.config.step_kind)

    def prng_key(self) -> PRNGKeyArray:
        return jax.random.PRNGKey(self.config.random_seed)

    def on_step_end(self, state: State) -> State:
        state = super().on_step_end(state)
        state.elapsed_time_s = time.time() - state.start_time_s
        return state

    def log_train_step(self, model: PyTree, batch: Batch, output: Output, state: State) -> None:
        """Override this function to do logging during the training phase.

        This function is called after the model forward pass and before the
        backward pass. It is called in the training phase.

        Args:
            model: The current model.
            batch: The batch from the dataloader.
            output: The model output.
            state: The current training state.
        """

    def log_valid_step(self, model: PyTree, batch: Batch, output: Output, state: State) -> None:
        """Override this function to do logging during the validation phase.

        This function is called after the model forward pass. It is called in
        the validation phase.

        Args:
            model: The current model.
            batch: The batch from the dataloader.
            output: The model output.
            state: The current training state.
        """

    def log_state_timers(self, state: State) -> None:
        timer = self.state_timers[state.phase]
        timer.step(state)
        for ns, d in timer.log_dict().items():
            for k, v in d.items():
                self.logger.log_scalar(k, v, namespace=ns)

    def log_step(self, model: PyTree, batch: Batch, output: Output, loss: Array, state: State) -> None:
        phase = state.phase

        self.logger.log_scalar("loss", loss, namespace="loss")
        self.log_state_timers(state)

        # Delegate to the appropriate logging function based on the phase.
        match phase:
            case "train":
                self.log_train_step(model, batch, output, state)
            case "valid":
                self.log_valid_step(model, batch, output, state)
            case _:
                raise KeyError(f"Unknown phase: {phase}")

        self.write_logs(state)

    @abstractmethod
    def get_model(self, key: PRNGKeyArray) -> PyTree:
        """Returns the Equinox model to train.

        Returns:
            The model to train.
        """

    @abstractmethod
    def get_optimizer(self) -> optax.GradientTransformation:
        """Gets the optimizer for the model.

        Returns:
            The optimizer to use to train the model.
        """

    def get_initial_opt_state(self, model: PyTree, optimizer: optax.GradientTransformation) -> optax.OptState:
        return optimizer.init(eqx.filter(model, eqx.is_array))

    @overload
    def load_initial_state(
        self,
        key: PRNGKeyArray,
        load_optimizer: Literal[False] = False,
    ) -> tuple[PyTree, State]: ...

    @overload
    def load_initial_state(
        self,
        key: PRNGKeyArray,
        load_optimizer: Literal[True],
    ) -> tuple[PyTree, optax.GradientTransformation, optax.OptState, State]: ...

    def load_initial_state(
        self,
        key: PRNGKeyArray,
        load_optimizer: bool = False,
    ) -> tuple[PyTree, State] | tuple[PyTree, optax.GradientTransformation, optax.OptState, State]:
        init_ckpt_path = self.get_init_ckpt_path()

        if init_ckpt_path is not None:
            logger.info("Loading checkpoint from %s", init_ckpt_path)
            if load_optimizer:
                model, optimizer, opt_state, state, config = self.load_checkpoint(init_ckpt_path)
                config_diff = get_diff_string(diff_configs(config, cast(DictConfig, self.config)))
                if config_diff:
                    logger.warning("Loaded config differs from current config:\n%s", config_diff)
                return model, optimizer, opt_state, state

            else:
                model, state, config = self.load_checkpoint(init_ckpt_path, "model_state_config")
                config_diff = get_diff_string(diff_configs(config, cast(DictConfig, self.config)))
                if config_diff:
                    logger.warning("Loaded config differs from current config:\n%s", config_diff)
                return model, state

        model = self.get_model(key)
        state = State.init_state()

        if not load_optimizer:
            return model, state

        optimizer = self.get_optimizer()
        opt_state = self.get_initial_opt_state(model, optimizer)

        return model, optimizer, opt_state, state

    @eqx.filter_jit
    def get_output(self, model: PyTree, batch: Batch) -> Output:
        """Gets the output from the model.

        By default, we assume the model is a function that takes the batch as
        input and returns the loss. This function can be patched to do more
        complex operations instead.

        Args:
            model: The current model.
            batch: The current minibatch of samples.
        """
        raise NotImplementedError("`get_output` must be implemented by the subclass")

    @eqx.filter_jit
    def compute_loss(self, model: PyTree, batch: Batch, output: Output) -> Array:
        """Gets the loss for the current batch.

        By default, we assume the model is a function that takes the batch as
        input and returns the loss. This function can be patched to do more
        complex operations instead.

        Args:
            model: The current model.
            batch: The current minibatch of samples.
            output: The output from the model.

        Returns:
            The computed loss, as a tensor.
        """
        if not isinstance(output, Array):
            raise ValueError(f"When model output is not the loss, you must override `compute_loss`. Got {type(output)}")
        return output

    @eqx.filter_jit
    def get_output_and_loss(self, model: PyTree, batch: Batch) -> tuple[Array, Output]:
        output = self.get_output(model, batch)
        loss = self.compute_loss(model, batch, output)
        return loss, output

    @eqx.filter_jit
    def update(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        batch: Batch,
    ) -> tuple[Array, PyTree, optax.OptState, Output]:
        (loss, output), grads = eqx.filter_value_and_grad(self.get_output_and_loss, has_aux=True)(model, batch)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, output

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
        self._training_over_flag = True

    def maybe_log_termination_time(self, remaining_percent: float, state: State) -> None:
        if self._last_printed_remaining_time + PRINT_FINISH_TIME_EVERY_N_SECONDS > state.elapsed_time_s:
            return
        self._last_printed_remaining_time = state.elapsed_time_s
        remaining_seconds = remaining_percent * state.elapsed_time_s / (1 - remaining_percent)
        termination_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining_seconds))
        # logger.info("Estimated finish time: %s", termination_time)
        jax.debug.print("Estimated finish time: {}", termination_time)

    def get_remaining_percent(self, state: State) -> float | None:
        if self.config.max_steps is None:
            return None
        return (self.config.max_steps - self.get_step(state)) / self.config.max_steps

    def is_training_over(self, state: State) -> bool:
        if self._training_over_flag:
            return True
        remaining_percent = self.get_remaining_percent(state)
        if remaining_percent is None:
            return False
        self.maybe_log_termination_time(remaining_percent, state)
        return remaining_percent <= 0.0

    def get_step(self, state: State) -> int:
        match self._step_kind:
            case "step":
                return state.num_steps
            case "sample":
                return state.num_samples
            case "second":
                return int(state.elapsed_time_s)
            case _:
                raise ValueError(f"Invalid step kind {self._step_kind}")

    def log_state(self) -> None:
        logger.log(LOG_STATUS, self.task_path)
        logger.log(LOG_STATUS, self.task_name)
        logger.log(LOG_STATUS, "JAX devices: %s", jax.devices())
        self.logger.log_file("git_state.txt", get_git_state(self))
        self.logger.log_file("training_code.txt", get_training_code(self))
        self.logger.log_file("config.yaml", self.config_str(self.config, use_cli=False))

    @eqx.filter_jit
    def train_step(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        batch: Batch,
    ) -> tuple[PyTree, optax.OptState, Array, Output]:
        loss, model, opt_state, output = self.update(model, optimizer, opt_state, batch)
        return model, opt_state, loss, output

    @eqx.filter_jit
    def val_step(self, model: PyTree, batch: Batch) -> tuple[PyTree, Array, Output]:
        loss, output = eqx.filter_jit(self.get_output_and_loss)(model, batch)
        return model, loss, output

    def train_loop(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        train_pf: Iterator[Batch],
        valid_pf: Iterator[Batch],
        state: State,
    ) -> None:
        while not self.is_training_over(state):
            if self.valid_step_timer.is_valid_step(state):
                valid_batch = next(valid_pf)
                with self.step_context("model_step"):
                    model, loss, output = self.val_step(model, valid_batch)

                # Perform logging.
                with self.step_context("write_logs"):
                    state.phase = "valid"
                    self.log_step(model, valid_batch, output, loss, state)
                    state.num_valid_samples += 1

            state = self.on_step_start(state)

            with self.step_context("model_step"):
                train_batch = next(train_pf)
                model, opt_state, loss, output = self.train_step(model, optimizer, opt_state, train_batch)

            with self.step_context("write_logs"):
                state.phase = "train"
                self.log_step(model, train_batch, output, loss, state)
                state.num_steps += 1
                state.num_samples += self.get_size_of_batch(train_batch) or 0

            state = self.on_step_end(state)

            if self.should_checkpoint(state):
                self.save_checkpoint(model, optimizer, opt_state, state)

        # After finishing training, save the final checkpoint.
        self.save_checkpoint(model, optimizer, opt_state, state)

    @contextlib.contextmanager
    def get_train_iterator(self) -> Generator[Iterator[Batch], None, None]:
        try:
            train_iterator: Iterator[Batch] = self.get_data_iterator("train")
            yield train_iterator
            return
        except NotImplementedError:
            pass

        train_ds = self.get_dataset("train")
        train_dl = self.get_dataloader(train_ds, "train")
        train_pf = self.get_prefetcher(train_dl)

        try:
            with train_pf as train_pf_ctx:
                yield train_pf_ctx
        finally:
            logger.info("Closing train prefetcher")

    @contextlib.contextmanager
    def get_valid_iterator(self) -> Generator[Iterator[Batch], None, None]:
        try:
            valid_iterator: Iterator[Batch] = self.get_data_iterator("valid")
            yield valid_iterator
            return
        except NotImplementedError:
            pass

        valid_ds = self.get_dataset("valid")
        valid_dl = self.get_dataloader(valid_ds, "valid")
        valid_pf = self.get_prefetcher(valid_dl)

        try:
            with valid_pf as valid_pf_ctx:
                yield valid_pf_ctx
        finally:
            logger.info("Closing valid prefetcher")

    def run(self) -> None:
        self.run_training()

    def run_training(self) -> None:
        """Runs the training loop.

        Args:
            model: The current model
            task: The current task
            optimizer: The current optimizer
            lr_scheduler: The current learning rate scheduler

        Raises:
            ValueError: If the task is not a supervised learning task
        """
        with self:
            key = self.prng_key()

            self.set_loggers()

            if is_master():
                Thread(target=self.log_state, daemon=True).start()

            key, model_key = jax.random.split(key)
            model, optimizer, opt_state, state = self.load_initial_state(model_key, load_optimizer=True)
            state = self.on_training_start(state)

            def on_exit() -> None:
                self.save_checkpoint(model, optimizer, opt_state, state)

            # Handle user-defined interrupts during the training loop.
            self.add_signal_handler(on_exit, signal.SIGUSR1, signal.SIGTERM)

            with self.get_train_iterator() as train_pf, self.get_valid_iterator() as valid_pf:
                try:
                    self.train_loop(
                        model=model,
                        optimizer=optimizer,
                        opt_state=opt_state,
                        train_pf=train_pf,
                        valid_pf=valid_pf,
                        state=state,
                    )

                except TrainingFinishedError:
                    if is_master():
                        show_info(
                            f"Finished training after {state.num_steps} steps, {state.num_samples} samples",
                            important=True,
                        )
                    self.save_checkpoint(model, optimizer, opt_state, state)

                except (KeyboardInterrupt, bdb.BdbQuit):
                    if is_master():
                        show_info("Interrupted training", important=True)

                except BaseException:
                    exception_tb = textwrap.indent(highlight_exception_message(traceback.format_exc()), "  ")
                    sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                    sys.stdout.flush()
                    self.save_checkpoint(model, optimizer, opt_state, state)

                finally:
                    state = self.on_training_end(state)
