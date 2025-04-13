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
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
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
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

from xax.core.conf import field
from xax.core.state import Phase, State
from xax.nn.functions import set_random_seed
from xax.nn.parallel import is_master
from xax.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin
from xax.task.mixins.checkpointing import CheckpointingConfig, CheckpointingMixin, CheckpointPart
from xax.task.mixins.data_loader import DataloadersConfig, DataloadersMixin
from xax.task.mixins.logger import LoggerConfig, LoggerMixin
from xax.task.mixins.runnable import RunnableConfig, RunnableMixin
from xax.task.mixins.step_wrapper import StepContextConfig, StepContextMixin
from xax.utils.experiments import (
    ContextTimer,
    StateTimer,
    TrainingFinishedError,
    diff_configs,
    get_diff_string,
    get_info_json,
    get_state_file_string,
    get_training_code,
)
from xax.utils.jax import jit as xax_jit
from xax.utils.logging import LOG_PING, LOG_STATUS
from xax.utils.text import highlight_exception_message, show_info
from xax.utils.types.frozen_dict import FrozenDict

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
            self.last_valid_time = state.elapsed_time_s.item()
            self.last_valid_step = state.num_steps.item()
            return False

        # Step-based validation.
        valid_every_n_steps = self.valid_every_n_steps
        if valid_every_n_steps is not None and state.num_steps >= valid_every_n_steps + self.last_valid_step:
            self.last_valid_step = state.num_steps.item()
            return True

        # Time-based validation.
        valid_every_n_seconds = self.valid_every_n_seconds
        if (
            valid_every_n_seconds is not None
            and state.elapsed_time_s.item() - self.last_valid_time >= valid_every_n_seconds
        ):
            self.last_valid_time = state.elapsed_time_s.item()
            return True

        # Time-based validation for first validation step.
        if self.first_valid_step_flag:
            valid_first_n_seconds = self.valid_first_n_seconds
            if valid_first_n_seconds is not None and state.elapsed_time_s.item() >= valid_first_n_seconds:
                self.last_valid_time = state.elapsed_time_s.item()
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
    global_grad_clip: float = field(value=10.0, help="The maximum gradient norm to clip to.")


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

    def log_train_step(
        self,
        model: PyTree,
        batch: Batch,
        output: Output,
        metrics: FrozenDict[str, Array],
        state: State,
    ) -> None:
        """Override this function to do logging during the training phase.

        This function is called after the model forward pass and before the
        backward pass. It is called in the training phase.

        Args:
            model: The current model.
            batch: The batch from the dataloader.
            output: The model output.
            metrics: The metrics for the current batch.
            state: The current training state.
        """

    def log_valid_step(
        self,
        model: PyTree,
        batch: Batch,
        output: Output,
        metrics: FrozenDict[str, Array],
        state: State,
    ) -> None:
        """Override this function to do logging during the validation phase.

        This function is called after the model forward pass. It is called in
        the validation phase.

        Args:
            model: The current model.
            batch: The batch from the dataloader.
            output: The model output.
            metrics: The metrics for the current batch.
            state: The current training state.
        """

    def log_state_timers(self, state: State) -> None:
        timer = self.state_timers[state.phase]
        timer.step(state)
        for k, v in timer.log_dict().items():
            if isinstance(v, tuple):
                v, secondary = v
            else:
                secondary = False
            self.logger.log_scalar(k, v, namespace="⌛ timers", secondary=secondary)

    def log_step(
        self,
        model: PyTree,
        batch: Batch,
        output: Output,
        metrics: FrozenDict[str, Array],
        state: State,
    ) -> None:
        phase = state.phase

        for k, v in metrics.items():
            if v.size == 1:
                self.logger.log_scalar(k, v.item())
            else:
                self.logger.log_histogram(k, v)

        self.log_state_timers(state)

        # Delegate to the appropriate logging function based on the phase.
        match phase:
            case "train":
                self.log_train_step(model, batch, output, metrics, state)
            case "valid":
                self.log_valid_step(model, batch, output, metrics, state)
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
            model, state, config = self.load_ckpt(init_ckpt_path, part="model_state_config")
            config_diff = get_diff_string(diff_configs(asdict(config), asdict(self.config)))
            if config_diff:
                logger.warning("Loaded config differs from current config:\n%s", config_diff)

            if not load_optimizer:
                return model, state

            optimizer = self.load_ckpt(init_ckpt_path, part="opt")
            opt_state = self.load_ckpt(init_ckpt_path, part="opt_state", model=model, optimizer=optimizer)
            return model, optimizer, opt_state, state

        logger.info("Starting a new training run")
        model = self.get_model(key)
        state = State.init_state()

        if not load_optimizer:
            return model, state

        optimizer = self.get_optimizer()
        opt_state = self.get_initial_opt_state(model, optimizer)

        return model, optimizer, opt_state, state

    @overload
    def load_ckpt(
        self,
        path: Path,
        *,
        part: Literal["all"],
    ) -> tuple[PyTree, optax.GradientTransformation, optax.OptState, State, Config]: ...

    @overload
    def load_ckpt(
        self,
        path: Path,
        *,
        part: Literal["model_state_config"],
    ) -> tuple[PyTree, State, Config]: ...

    @overload
    def load_ckpt(
        self,
        path: Path,
        *,
        part: Literal["model"],
    ) -> PyTree: ...

    @overload
    def load_ckpt(
        self,
        path: Path,
        *,
        part: Literal["opt"],
    ) -> optax.GradientTransformation: ...

    @overload
    def load_ckpt(
        self,
        path: Path,
        *,
        part: Literal["opt_state"],
        model: PyTree | None = None,
        optimizer: optax.GradientTransformation | None = None,
    ) -> optax.OptState: ...

    @overload
    def load_ckpt(
        self,
        path: Path,
        *,
        part: Literal["state"],
    ) -> State: ...

    @overload
    def load_ckpt(
        self,
        path: Path,
        *,
        part: Literal["config"],
    ) -> Config: ...

    def load_ckpt(
        self,
        path: str | Path,
        *,
        part: CheckpointPart = "all",
        model: PyTree | None = None,
        optimizer: optax.GradientTransformation | None = None,
    ) -> (
        tuple[PyTree, optax.GradientTransformation, optax.OptState, State, Config]
        | tuple[PyTree, State, Config]
        | PyTree
        | optax.GradientTransformation
        | optax.OptState
        | State
        | Config
    ):
        path = Path(path)

        # This key isn't used for anything, it's just a required argument.
        key = jax.random.PRNGKey(0)

        match part:
            case "model_state_config":
                model_spec = eqx.filter_eval_shape(self.get_model, key)
                return self.load_ckpt_with_template(path, part="model_state_config", model_template=model_spec)

            case "model":
                model_spec = eqx.filter_eval_shape(self.get_model, key)
                return self.load_ckpt_with_template(path, part="model", model_template=model_spec)

            case "config":
                return self.load_ckpt_with_template(path, part="config")

            case "opt":
                optimizer_spec = eqx.filter_eval_shape(self.get_optimizer)
                return self.load_ckpt_with_template(path, part="opt", optimizer_template=optimizer_spec)

            case "opt_state":
                if model is None:
                    model_spec = eqx.filter_eval_shape(self.get_model, key)
                    model = self.load_ckpt_with_template(path, part="model", model_template=model_spec)
                if optimizer is None:
                    optimizer_spec = eqx.filter_eval_shape(self.get_optimizer)
                    optimizer = self.load_ckpt_with_template(path, part="opt", optimizer_template=optimizer_spec)
                opt_state_spec = eqx.filter_eval_shape(self.get_initial_opt_state, model, optimizer)
                return self.load_ckpt_with_template(path, part="opt_state", opt_state_template=opt_state_spec)

            case "state":
                return self.load_ckpt_with_template(path, part="state")

            case "config":
                return self.load_ckpt_with_template(path, part="config")

            case "all":
                model_spec = eqx.filter_eval_shape(self.get_model, key)
                model = self.load_ckpt_with_template(path, part="model", model_template=model_spec)
                optimizer_spec = eqx.filter_eval_shape(self.get_optimizer)
                optimizer = self.load_ckpt_with_template(path, part="opt", optimizer_template=optimizer_spec)
                opt_state_spec = eqx.filter_eval_shape(self.get_initial_opt_state, model, optimizer)
                opt_state = self.load_ckpt_with_template(path, part="opt_state", opt_state_template=opt_state_spec)
                state = self.load_ckpt_with_template(path, part="state")
                config = self.load_ckpt_with_template(path, part="config")
                return model, optimizer, opt_state, state, config

            case _:
                raise ValueError(f"Unknown checkpoint part: {part}")

    def get_output(self, model: PyTree, batch: Batch, state: State) -> Output:
        """Gets the output from the model.

        By default, we assume the model is a function that takes the batch as
        input and returns the loss. This function can be patched to do more
        complex operations instead.

        Args:
            model: The current model.
            batch: The current minibatch of samples.
            state: The current training state.
        """
        raise NotImplementedError("`get_output` must be implemented by the subclass")

    def compute_loss(self, model: PyTree, batch: Batch, output: Output, state: State) -> Array:
        """Gets the loss for the current batch.

        By default, we assume the model is a function that takes the batch as
        input and returns the loss. This function can be patched to do more
        complex operations instead.

        Args:
            model: The current model.
            batch: The current minibatch of samples.
            output: The output from the model.
            state: The current training state.

        Returns:
            The computed loss, as a tensor.
        """
        if not isinstance(output, Array):
            raise ValueError(f"When model output is not the loss, you must override `compute_loss`. Got {type(output)}")
        return output

    def compute_metrics(
        self,
        model: PyTree,
        batch: Batch,
        output: Output,
        loss: Array,
        state: State,
    ) -> dict[str, Array]:
        """Computes the metrics for the current batch.

        Args:
            model: The current model.
            batch: The current minibatch of samples.
            output: The output from the model.
            loss: The loss for the current batch.
            state: The current training state.

        Returns:
            A dictionary of metrics.
        """
        return {
            "loss": loss,
        }

    @xax_jit(static_argnames=["self", "model_static"], jit_level=3)
    def get_output_and_loss(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        batch: Batch,
        state: State,
    ) -> tuple[Array, tuple[Output, dict[str, Array]]]:
        model = eqx.combine(model_arr, model_static)
        output = self.get_output(model, batch, state)
        loss = self.compute_loss(model, batch, output, state)
        metrics = self.compute_metrics(model, batch, output, loss, state)
        return loss, (output, metrics)

    def update(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        batch: Batch,
        state: State,
    ) -> tuple[PyTree, optax.OptState, Output, dict[str, Array]]:
        grad_fn = jax.grad(self.get_output_and_loss, argnums=0, has_aux=True)
        grad_fn = xax_jit(static_argnums=[1], jit_level=3)(grad_fn)
        grads, (output, metrics) = grad_fn(model_arr, model_static, batch, state)
        model_arr, opt_state, grad_metrics = self.apply_gradients_with_clipping(model_arr, grads, optimizer, opt_state)
        return model_arr, opt_state, output, metrics | grad_metrics

    @xax_jit(static_argnames=["self", "optimizer"], jit_level=3)
    def apply_gradients_with_clipping(
        self,
        model_arr: PyTree,
        grads: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
    ) -> tuple[PyTree, optax.OptState, dict[str, Array]]:
        grad_norm = optax.global_norm(grads)
        grad_metrics = {"grad_norm": grad_norm}

        def apply(grads: PyTree, grad_norm: Array) -> tuple[PyTree, optax.OptState]:
            # Clip the global gradient norm to some desired range.
            grad_factor = self.config.global_grad_clip / jnp.maximum(grad_norm, 1e-6)
            grads = jax.tree.map(lambda x: x * grad_factor, grads)

            # Apply the gradient updates.
            updates, new_opt_state = optimizer.update(grads, opt_state, model_arr)
            new_model_arr = eqx.apply_updates(model_arr, updates)
            return new_model_arr, new_opt_state

        # Don't apply updates if the gradient is NaN or Inf.
        new_model_arr, new_opt_state = jax.lax.cond(
            jnp.isnan(grad_norm) | jnp.isinf(grad_norm),
            lambda *_: (model_arr, opt_state),
            apply,
            grads,
            grad_norm,
        )

        return new_model_arr, new_opt_state, grad_metrics

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
        self._last_printed_remaining_time = state.elapsed_time_s.item()
        remaining_seconds = remaining_percent * state.elapsed_time_s.item() / (1 - remaining_percent)
        termination_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining_seconds))
        logger.log(LOG_PING, "Estimated finish time: %s", termination_time)

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
                return int(state.num_steps.item())
            case "sample":
                return int(state.num_samples.item())
            case "second":
                return int(state.elapsed_time_s.item())
            case _:
                raise ValueError(f"Invalid step kind {self._step_kind}")

    def log_state(self) -> None:
        logger.log(LOG_STATUS, self.task_path)
        logger.log(LOG_STATUS, self.task_name)
        logger.log(LOG_STATUS, "JAX devices: %s", jax.devices())
        self.logger.log_file("state.txt", get_state_file_string(self))
        self.logger.log_file("training_code.py", get_training_code(self))
        self.logger.log_file("config.yaml", self.config_str(self.config, use_cli=False))
        self.logger.log_file("info.json", get_info_json())

    def model_partition_fn(self, item: Any) -> bool:  # noqa: ANN401
        return eqx.is_inexact_array(item)

    @xax_jit(static_argnames=["self", "model_static", "optimizer"], jit_level=3)
    def train_step(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        batch: Batch,
        state: State,
    ) -> tuple[PyTree, optax.OptState, Output, FrozenDict[str, Array]]:
        model_arr, opt_state, output, metrics = self.update(model_arr, model_static, optimizer, opt_state, batch, state)
        return model_arr, opt_state, output, FrozenDict(metrics)

    @xax_jit(static_argnames=["self", "model_static"], jit_level=3)
    def val_step(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        batch: Batch,
        state: State,
    ) -> tuple[Output, FrozenDict[str, Array]]:
        _, (output, metrics) = self.get_output_and_loss(model_arr, model_static, batch, state)
        return output, FrozenDict(metrics)

    def train_loop(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        train_pf: Iterator[Batch],
        valid_pf: Iterator[Batch],
        state: State,
    ) -> None:
        model_arr, model_static = eqx.partition(model, self.model_partition_fn)

        while not self.is_training_over(state):
            if self.valid_step_timer.is_valid_step(state):
                with ContextTimer() as timer:
                    valid_batch = next(valid_pf)
                    output, metrics = self.val_step(model_arr, model_static, valid_batch, state)
                    self.log_step(eqx.combine(model_arr, model_static), valid_batch, output, metrics, state)

                state = state.replace(
                    phase="valid",
                    num_valid_steps=state.num_valid_steps + 1,
                    num_valid_samples=state.num_valid_samples + (self.get_size_of_batch(valid_batch) or 0),
                    valid_elapsed_time_s=state.valid_elapsed_time_s + timer.elapsed_time,
                )

            with ContextTimer() as timer:
                state = self.on_step_start(state)
                train_batch = next(train_pf)
                model_arr, opt_state, output, metrics = self.train_step(
                    model_arr=model_arr,
                    model_static=model_static,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    batch=train_batch,
                    state=state,
                )
                self.log_step(eqx.combine(model_arr, model_static), train_batch, output, metrics, state)

            state = state.replace(
                phase="train",
                num_steps=state.num_steps + 1,
                num_samples=state.num_samples + (self.get_size_of_batch(train_batch) or 0),
                elapsed_time_s=state.elapsed_time_s + timer.elapsed_time,
            )

            state = self.on_step_end(state)

            if self.should_checkpoint(state):
                model = eqx.combine(model_arr, model_static)
                self.save_checkpoint(model=model, optimizer=optimizer, opt_state=opt_state, state=state)

        # After finishing training, save the final checkpoint.
        model = eqx.combine(model_arr, model_static)
        self.save_checkpoint(model=model, optimizer=optimizer, opt_state=opt_state, state=state)

    @contextlib.contextmanager
    def get_train_iterator(self, key: PRNGKeyArray) -> Generator[Iterator[Batch], None, None]:
        try:
            train_iterator: Iterator[Batch] = self.get_data_iterator("train", key=key)
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
    def get_valid_iterator(self, key: PRNGKeyArray) -> Generator[Iterator[Batch], None, None]:
        try:
            valid_iterator: Iterator[Batch] = self.get_data_iterator("valid", key=key)
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
                self.save_checkpoint(model=model, optimizer=optimizer, opt_state=opt_state, state=state)

            # Handle user-defined interrupts during the training loop.
            self.add_signal_handler(on_exit, signal.SIGUSR1, signal.SIGTERM)

            key, tkey, vkey = jax.random.split(key, 3)
            with self.get_train_iterator(tkey) as train_pf, self.get_valid_iterator(vkey) as valid_pf:
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
                        num_steps, num_samples = int(state.num_steps), int(state.num_samples)
                        show_info(f"Finished training after {num_steps} steps, {num_samples} samples", important=True)
                    self.save_checkpoint(model=model, optimizer=optimizer, opt_state=opt_state, state=state)

                except (KeyboardInterrupt, bdb.BdbQuit):
                    if is_master():
                        show_info("Interrupted training", important=True)

                except BaseException:
                    exception_tb = textwrap.indent(highlight_exception_message(traceback.format_exc()), "  ")
                    sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                    sys.stdout.flush()
                    self.save_checkpoint(model=model, optimizer=optimizer, opt_state=opt_state, state=state)

                finally:
                    state = self.on_training_end(state)
