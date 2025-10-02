"""Defines a mixin for JMP (JAX Mixed Precision) support in training tasks."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import optax
from jaxtyping import Array, PyTree

from xax.core.conf import field
from xax.core.state import State
from xax.task.base import BaseConfig, BaseTask
from xax.task.mixins.train import Batch, InitParams, Output
from xax.utils.types.frozen_dict import FrozenDict

logger = logging.getLogger(__name__)


class PrecisionPolicy(Enum):
    """Mixed precision policy options for JMP."""

    FULL = "full"
    HALF_PARAM = "half_param"
    HALF_COMPUTE = "half_compute"
    HALF_OUTPUT = "half_output"
    HALF_FRONT = "half_front"
    HALF_BACK = "half_back"
    HALF_TOTAL = "half_total"
    BF_PARAM = "bf_param"
    BF_COMPUTE = "bf_compute"
    BF_OUTPUT = "bf_output"
    BF_FRONT = "bf_front"
    BF_BACK = "bf_back"
    BF_TOTAL = "bf_total"


@jax.tree_util.register_dataclass
@dataclass
class MixedPrecisionConfig(BaseConfig):
    """Updated configuration class to include mixed precision training."""

    # JMP Policy Configuration
    precision_policy: PrecisionPolicy = field(PrecisionPolicy.FULL, help="Mixed precision policy for training")

    # Loss Scaling Configuration
    enable_loss_scaling: bool = field(True, help="Enable loss scaling for half precision training")
    initial_loss_scale: float = field(2**15, help="Initial loss scale value for dynamic scaling")
    dynamic_period: int = field(2000, help="Period for dynamic loss scaling adjustment")
    dynamic_factor: float = field(2.0, help="Factor for dynamic loss scaling adjustment")
    min_loss_scale: float = field(1.0, help="Minimum loss scale value")

    # Gradient Handling
    skip_nonfinite_updates: bool = field(True, help="Skip optimizer updates when gradients are non-finite")

    # Loss Scale Monitoring
    log_loss_scale_every_n_steps: int = field(1000, help="Log current loss scale every N steps (0 to disable)")


Config = TypeVar("Config", bound=MixedPrecisionConfig)


class MixedPrecisionMixin(BaseTask[Config], Generic[Config]):
    """Applies mixed precision to either input data, computation, and/or layer outputs.

    Dynamic loss scaling also added as an option to handle potential
    underflow in half-precision training.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Initialize JMP components based on config
        self.mp_policy: jmp.Policy = self._create_mixed_precision_policy()
        self.loss_scaler: jmp.LossScale = self._create_loss_scaler()

    def _create_mixed_precision_policy(self) -> jmp.Policy:
        match self.config.precision_policy:
            case PrecisionPolicy.FULL:
                return jmp.get_policy("params=float32,compute=float32,output=float32")
            case PrecisionPolicy.HALF_PARAM:
                return jmp.get_policy("params=float16,compute=float16,output=float32")
            case PrecisionPolicy.HALF_COMPUTE:
                return jmp.get_policy("params=float32,compute=float16,output=float32")
            case PrecisionPolicy.HALF_OUTPUT:
                return jmp.get_policy("params=float32,compute=float32,output=float16")
            case PrecisionPolicy.HALF_FRONT:
                return jmp.get_policy("params=float16,compute=float16,output=float32")
            case PrecisionPolicy.HALF_BACK:
                return jmp.get_policy("params=float32,compute=float16,output=float16")
            case PrecisionPolicy.HALF_TOTAL:
                return jmp.get_policy("params=float16,compute=float16,output=float16")
            case PrecisionPolicy.BF_PARAM:
                return jmp.get_policy("params=bfloat16,compute=float32,output=float32")
            case PrecisionPolicy.BF_COMPUTE:
                return jmp.get_policy("params=float32,compute=bfloat16,output=float32")
            case PrecisionPolicy.BF_OUTPUT:
                return jmp.get_policy("params=float32,compute=float32,output=bfloat16")
            case PrecisionPolicy.BF_FRONT:
                return jmp.get_policy("params=bfloat16,compute=bfloat16,output=float32")
            case PrecisionPolicy.BF_BACK:
                return jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
            case PrecisionPolicy.BF_TOTAL:
                return jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
            case _:
                raise RuntimeError(
                    f"Unhandled precision policy: {self.config.precision_policy}. "
                    f"This message shouldn't be seen (should already be captured by OmegaConf)."
                )

    def _create_loss_scaler(self) -> jmp.LossScale:
        """Dynamics loss scaler implementation."""
        if self.config.precision_policy.name.startswith("BF_") or not self.config.enable_loss_scaling:
            return jmp.NoOpLossScale()

        return jmp.DynamicLossScale(
            loss_scale=jnp.array(self.config.initial_loss_scale, dtype=jnp.float32),
            period=self.config.dynamic_period,
            factor=self.config.dynamic_factor,
            min_loss_scale=jnp.array(self.config.min_loss_scale, dtype=jnp.float32),
        )

    def get_model(self, params: InitParams) -> PyTree:
        """Adds mixed precision support to model parameters."""
        # Get base model from parent mixin
        model = super().get_model(params)  # type: ignore[misc]

        # Convert model parameters to precision policy
        model_casted = self.mp_policy.cast_to_param(model)

        logger.debug("Model parameters cast to precision: %s", self.mp_policy.param_dtype)
        return model_casted

    def get_output_and_loss(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        batch: Batch,
        state: State,
    ) -> tuple[Array, tuple[Output, dict[str, Array]]]:
        """Mixed precision support to forward pass."""
        # Convert input data and model weights to precision policy
        batch_casted = self.mp_policy.cast_to_compute(batch)
        model_arr_compute = self.mp_policy.cast_to_compute(model_arr)
        model_static_compute = self.mp_policy.cast_to_compute(model_static)

        # Call original method (from SupervisedTask.get_output_and_loss() ) on new FP16 data from above
        loss, (output, metrics) = super().get_output_and_loss(  # type: ignore[misc]
            model_arr_compute, model_static_compute, batch_casted, state
        )

        # Multiply loss by large factor to prevent gradient underflow
        scaled_loss = self.loss_scaler.scale(loss)

        # Convert result back to higher precision
        output_casted = self.mp_policy.cast_to_output(output)
        metrics_casted = {k: self.mp_policy.cast_to_output(v) for k, v in metrics.items()}

        return scaled_loss, (output_casted, metrics_casted)

    def update(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        batch: Batch,
        state: State,
    ) -> tuple[PyTree, optax.OptState, Output, dict[str, Array]]:
        """Handles gradient loss scaling and unscaling."""
        # Compute gradients using scaled loss
        grad_fn = jax.grad(self.get_output_and_loss, argnums=0, has_aux=True)
        grads, (output, metrics) = grad_fn(model_arr, model_static, batch, state)

        # Unscale gradients
        grads = self.loss_scaler.unscale(grads)

        # Check if gradients are finite
        grads_finite = jmp.all_finite(grads)

        # Adjust loss scale for dynamic scaling (depends on config)
        self.loss_scaler = self.loss_scaler.adjust(grads_finite)

        # Update parameters only if gradients are finite
        def apply_updates(carry: tuple[PyTree, optax.OptState]) -> tuple[PyTree, optax.OptState]:
            model, opt_st = carry
            updates, new_opt_state = optimizer.update(grads, opt_st, model)
            updated_model = eqx.apply_updates(model, updates)
            return updated_model, new_opt_state

        def skip_updates(carry: tuple[PyTree, optax.OptState]) -> tuple[PyTree, optax.OptState]:
            return carry

        should_update = jnp.logical_or(grads_finite, jnp.logical_not(self.config.skip_nonfinite_updates))
        model_arr, opt_state = jax.lax.cond(should_update, apply_updates, skip_updates, (model_arr, opt_state))

        return model_arr, opt_state, output, metrics

    def val_step(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        batch: Batch,
        state: State,
    ) -> tuple[Output, FrozenDict[str, Array]]:
        """Adjusted validation step to support mixed precision.

        Applies the same precision casting as training but without gradient computation.
        """
        # Cast validation batch to compute precision
        batch_casted = self.mp_policy.cast_to_compute(batch)

        # Run forward pass (no scaling needed since no gradients)
        _, (output, metrics) = super().get_output_and_loss(model_arr, model_static, batch_casted, state)  # type: ignore[misc]

        # Cast outputs to output precision
        output_casted = self.mp_policy.cast_to_output(output)
        metrics_casted = {k: self.mp_policy.cast_to_output(v) for k, v in metrics.items()}

        return output_casted, FrozenDict(metrics_casted)

    def get_current_loss_scale(self) -> Array:
        """Helper function to get current loss scale value for tuning dynamic loss scaling."""
        return self.loss_scaler.loss_scale
