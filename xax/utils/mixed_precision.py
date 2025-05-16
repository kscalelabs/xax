"""Mixed precision training utilities for XAX.

This module provides utilities for mixed precision training in JAX, inspired by
the DeepMind JMP library (https://github.com/google-deepmind/jmp).

Mixed precision training is a technique that uses lower precision data types
(like float16 or bfloat16) during computation to speed up training, while
maintaining model weights in higher precision (like float32) to preserve
accuracy.
"""

import functools
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util

# Set XLA flags to optimize mixed precision performance
# These flags help with kernel fusion and performance on both GPUs and TPUs
# Only set them if they haven't been set already
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_fast_min_max=true "
        "--xla_gpu_enable_cublaslt=true "  # For newer NVIDIA GPUs
        # "--xla_tpu_enable_bfloat16_mmt=true "  # Remove unsupported flag
        # "--xla_gpu_enable_triton_softmax_fusion=true "  # Remove unsupported flag
    )
elif not any(flag in os.environ["XLA_FLAGS"] for flag in 
             ["xla_gpu_enable_fast_min_max", "xla_gpu_enable_cublaslt"]):
    os.environ["XLA_FLAGS"] += (
        " --xla_gpu_enable_fast_min_max=true"
        " --xla_gpu_enable_cublaslt=true"
        # " --xla_tpu_enable_bfloat16_mmt=true"  # Remove unsupported flag
        # " --xla_gpu_enable_triton_softmax_fusion=true"  # Remove unsupported flag
    )

from xax.utils.pytree import PyTree

# Type aliases for clarity
DType = jnp.dtype
T = TypeVar('T')
Params = PyTree


class Precision(str, Enum):
    """Precision types for mixed precision training."""
    HIGHEST = "highest"  # Use highest precision (typically float32)
    HIGH = "high"        # Use high precision (typically bfloat16 on TPU, float32 elsewhere)
    MEDIUM = "medium"    # Use medium precision (typically bfloat16)
    LOW = "low"          # Use low precision (typically float16)


def get_default_half_dtype() -> DType:
    """Return the default half-precision dtype for the current platform.
    
    Returns float16 for GPUs and bfloat16 for TPUs.
    """
    # Check if we're running on TPU
    if jax.devices()[0].platform == "tpu":
        return jnp.bfloat16
    return jnp.float16


def _parse_dtype(dtype_str: str) -> DType:
    """Parse a dtype from a string."""
    dtype_map = {
        "float16": jnp.float16,
        "float32": jnp.float32,
        "float64": jnp.float64,
        "bfloat16": jnp.bfloat16,
        "f16": jnp.float16,
        "f32": jnp.float32,
        "f64": jnp.float64,
        "bf16": jnp.bfloat16,
        "half": get_default_half_dtype(),
        "single": jnp.float32,
        "double": jnp.float64,
    }
    
    dtype_str = dtype_str.lower()
    if dtype_str in dtype_map:
        return dtype_map[dtype_str]
    
    raise ValueError(f"Unknown dtype: {dtype_str}")


@dataclass
class Policy:
    """A policy for mixed precision training.
    
    Attributes:
        param_dtype: Data type for parameters (typically float32).
        compute_dtype: Data type for computation (typically float16 or bfloat16).
        output_dtype: Data type for output (typically same as compute_dtype).
    """
    param_dtype: DType
    compute_dtype: DType
    output_dtype: DType
    
    @classmethod
    def from_string(cls, policy_str: str) -> "Policy":
        """Create a policy from a string.
        
        Example:
            Policy.from_string("params=float32,compute=float16,output=float16")
            Policy.from_string("float16")  # All in float16
        """
        if "," in policy_str:
            parts = {}
            for part in policy_str.split(","):
                key, value = part.split("=")
                parts[key.strip()] = value.strip()
            
            param_dtype = _parse_dtype(parts.get("params", "float32"))
            compute_dtype = _parse_dtype(parts.get("compute", "float16"))
            output_dtype = _parse_dtype(parts.get("output", parts.get("compute", "float16")))
        else:
            # Single dtype for everything
            dtype = _parse_dtype(policy_str)
            param_dtype = dtype
            compute_dtype = dtype
            output_dtype = dtype
        
        return cls(
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
        )
    
    def with_output_dtype(self, output_dtype: DType) -> "Policy":
        """Return a new policy with the specified output dtype."""
        return Policy(
            param_dtype=self.param_dtype,
            compute_dtype=self.compute_dtype,
            output_dtype=output_dtype,
        )
    
    def cast_to_compute(self, pytree: PyTree) -> PyTree:
        """Cast a pytree to the compute dtype."""
        return tree_map_dtype(self.compute_dtype, pytree)
    
    def cast_to_param(self, pytree: PyTree) -> PyTree:
        """Cast a pytree to the parameter dtype."""
        return tree_map_dtype(self.param_dtype, pytree)
    
    def cast_to_output(self, pytree: PyTree) -> PyTree:
        """Cast a pytree to the output dtype."""
        return tree_map_dtype(self.output_dtype, pytree)


def tree_map_dtype(dtype: DType, pytree: PyTree) -> PyTree:
    """Map a dtype over a pytree, casting all arrays to the specified dtype.
    
    Args:
        dtype: The target dtype.
        pytree: A JAX pytree.
    
    Returns:
        A new pytree with all arrays cast to the specified dtype.
    """
    def _cast_if_array(x):
        if isinstance(x, (jnp.ndarray, jax.Array)):
            return x.astype(dtype)
        return x
    
    return jax.tree.map(_cast_if_array, pytree)


# Common policies
DEFAULT = Policy(
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
    output_dtype=jnp.float32,
)

MIXED_PRECISION = Policy(
    param_dtype=jnp.float32,
    compute_dtype=get_default_half_dtype(),
    output_dtype=jnp.float32,
)

FULL_HALF = Policy(
    param_dtype=get_default_half_dtype(),
    compute_dtype=get_default_half_dtype(),
    output_dtype=get_default_half_dtype(),
)


def get_policy(policy_str: str) -> Policy:
    """Get a policy from a string.
    
    Args:
        policy_str: A string representation of the policy.
    
    Returns:
        A Policy object.
    """
    # Handle predefined policies
    if policy_str.lower() == "default":
        return DEFAULT
    elif policy_str.lower() == "mixed":
        return MIXED_PRECISION
    elif policy_str.lower() == "half" or policy_str.lower() == "float16":
        return FULL_HALF
    
    # Parse the string
    return Policy.from_string(policy_str)


class LossScale:
    """Base class for loss scaling strategies.
    
    Loss scaling is a technique used in mixed precision training where the loss
    is multiplied by a scaling factor before backpropagation. This helps prevent
    gradients from underflowing in reduced precision.
    """
    
    def scale(self, loss: jnp.ndarray) -> jnp.ndarray:
        """Scale the loss by the loss scale factor."""
        raise NotImplementedError
    
    def unscale(self, grads: PyTree) -> PyTree:
        """Unscale the gradients by the loss scale factor."""
        raise NotImplementedError
    
    def adjust(self, grads_finite: jnp.ndarray) -> "LossScale":
        """Adjust the loss scale based on whether gradients are finite."""
        return self
    
    @property
    def loss_scale(self) -> jnp.ndarray:
        """Get the current loss scale value."""
        raise NotImplementedError


class NoOpLossScale(LossScale):
    """A loss scale that does nothing.
    
    This is useful as a drop-in replacement when loss scaling is not needed.
    """
    
    def scale(self, loss: jnp.ndarray) -> jnp.ndarray:
        return loss
    
    def unscale(self, grads: PyTree) -> PyTree:
        return grads
    
    @property
    def loss_scale(self) -> jnp.ndarray:
        return jnp.array(1.0, dtype=jnp.float32)


class StaticLossScale(LossScale):
    """A static loss scale with a fixed value."""
    
    def __init__(self, scale: Union[float, jnp.ndarray]):
        """Initialize the static loss scale.
        
        Args:
            scale: The fixed loss scale value.
        """
        if isinstance(scale, float):
            scale = jnp.array(scale)
        self._scale = scale
    
    def scale(self, loss: jnp.ndarray) -> jnp.ndarray:
        """Scale the loss by the scale factor.
        
        Args:
            loss: The loss to scale.
            
        Returns:
            The scaled loss.
        """
        return loss * self._scale
    
    def unscale(self, grads: PyTree) -> PyTree:
        """Unscale the gradients by the scale factor.
        
        Args:
            grads: The gradients to unscale.
            
        Returns:
            The unscaled gradients.
        """
        return jax.tree.map(lambda g: g / self._scale, grads)
    
    @property
    def loss_scale(self) -> jnp.ndarray:
        """Get the current loss scale value."""
        return self._scale


class DynamicLossScale(LossScale):
    """A dynamic loss scale that adjusts itself during training.
    
    The scale is increased by a factor of 2 after a certain number of consecutive
    steps with finite gradients, and is decreased by a factor of 2 when non-finite
    gradients are encountered.
    """
    
    def __init__(
        self,
        initial_scale: Union[float, jnp.ndarray] = 2**15,
        growth_interval: int = 2000,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        max_scale: Optional[float] = None,
        dtype: DType = jnp.float32,
    ):
        """Initialize the dynamic loss scale.
        
        Args:
            initial_scale: The initial loss scale value.
            growth_interval: Number of consecutive steps with finite gradients before
                increasing the loss scale.
            growth_factor: Factor by which to increase the loss scale.
            backoff_factor: Factor by which to decrease the loss scale when non-finite
                gradients are encountered.
            max_scale: Maximum allowed loss scale value.
            dtype: Data type for the loss scale.
        """
        self._current_scale = jnp.array(initial_scale, dtype=dtype)
        self._growth_interval = growth_interval
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._max_scale = jnp.array(max_scale if max_scale is not None else float('inf'), dtype=dtype)
        self._steps_since_finite = 0
    
    def scale(self, loss: jnp.ndarray) -> jnp.ndarray:
        return loss * self._current_scale
    
    def unscale(self, grads: PyTree) -> PyTree:
        """Unscale the gradients by the current scale factor.
        
        Args:
            grads: The gradients to unscale.
            
        Returns:
            The unscaled gradients.
        """
        inv_scale = 1.0 / self._current_scale
        return jax.tree.map(lambda g: g * inv_scale, grads)
    
    def adjust(self, grads_finite: jnp.ndarray) -> "DynamicLossScale":
        """Adjust the loss scale based on whether gradients are finite.
        
        Args:
            grads_finite: A boolean indicating whether all gradients are finite.
        
        Returns:
            An updated DynamicLossScale instance.
        """
        # Create a new instance with updated values
        result = DynamicLossScale(
            initial_scale=self._current_scale,
            growth_interval=self._growth_interval,
            growth_factor=self._growth_factor,
            backoff_factor=self._backoff_factor,
            max_scale=self._max_scale,
            dtype=self._current_scale.dtype,
        )
        
        # Adjust the scale based on whether gradients are finite
        if grads_finite:
            result._steps_since_finite = self._steps_since_finite + 1
            if result._steps_since_finite >= self._growth_interval:
                new_scale = jnp.minimum(
                    self._current_scale * self._growth_factor,
                    self._max_scale
                )
                result._current_scale = new_scale
                result._steps_since_finite = 0
        else:
            result._current_scale = self._current_scale * self._backoff_factor
            result._steps_since_finite = 0
        
        return result
    
    @property
    def loss_scale(self) -> jnp.ndarray:
        return self._current_scale


def all_finite(pytree: PyTree) -> jnp.ndarray:
    """Check if all values in a pytree are finite.
    
    Args:
        pytree: The pytree to check.
        
    Returns:
        A boolean indicating whether all values are finite.
    """
    def _is_finite(x):
        if not isinstance(x, (jnp.ndarray, jax.Array)):
            return True
        return jnp.all(jnp.isfinite(x))
    
    finite_flags = jax.tree.map(_is_finite, pytree)
    return jax.tree_util.tree_reduce(jnp.logical_and, finite_flags, True)


def select_tree(pred: jnp.ndarray, a: PyTree, b: PyTree) -> PyTree:
    """Select between two pytrees based on a predicate.
    
    Args:
        pred: A boolean indicating which pytree to select.
        a: The pytree to select if pred is True.
        b: The pytree to select if pred is False.
        
    Returns:
        Either a or b depending on the value of pred.
    """
    return jax.tree.map(
        lambda a_val, b_val: jnp.where(pred, a_val, b_val), a, b
    )


def apply_mixed_precision(
    func: Callable[..., T],
    policy: Policy,
    static_argnums: Optional[Union[int, tuple[int, ...]]] = None
) -> Callable[..., T]:
    """Apply mixed precision to a function.
    
    This decorator applies the specified precision policy to a function's inputs
    and outputs.
    
    Args:
        func: The function to decorate.
        policy: The precision policy to apply.
        static_argnums: Indices of arguments that should not be cast.
    
    Returns:
        A decorated function that applies the precision policy.
    """
    if static_argnums is None:
        static_argnums = ()
    elif isinstance(static_argnums, int):
        static_argnums = (static_argnums,)
    
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # Cast args except for static args
        cast_args = []
        for i, arg in enumerate(args):
            if i in static_argnums:
                cast_args.append(arg)
            else:
                cast_args.append(policy.cast_to_compute(arg))
        
        # Cast kwargs
        cast_kwargs = {k: policy.cast_to_compute(v) for k, v in kwargs.items()}
        
        # Call the function
        result = func(*cast_args, **cast_kwargs)
        
        # Cast the result
        return policy.cast_to_output(result)
    
    return wrapped


# Performance monitoring utilities for mixed precision training

def compute_gradient_stats(grads: PyTree) -> Dict[str, jnp.ndarray]:
    """Compute statistics about gradients to help diagnose mixed precision issues.
    
    Args:
        grads: PyTree of gradients.
        
    Returns:
        Dictionary with gradient statistics like global norm, max, min, and 
        statistics about finite values.
    """
    # Convert any non-array leaves to arrays
    def preprocess(x):
        if not isinstance(x, (jnp.ndarray, jax.Array)):
            return jnp.array(0.0, dtype=jnp.float32)
        if not jnp.issubdtype(x.dtype, jnp.inexact):
            return jnp.array(0.0, dtype=jnp.float32)
        return x.astype(jnp.float32)  # Cast to float32 for stable computation
    
    # Get flattened list of arrays
    flat_grads = jax.tree_util.tree_leaves(jax.tree.map(preprocess, grads))
    if not flat_grads:  # If empty
        return {
            "grad_norm": jnp.array(0.0),
            "max_abs_grad": jnp.array(0.0),
            "min_abs_grad_nonzero": jnp.array(0.0),
            "has_nan": jnp.array(False),
            "has_inf": jnp.array(False),
            "finite_ratio": jnp.array(1.0),
        }
    
    # Compute global norm
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in flat_grads))
    
    # Compute max absolute gradient
    max_abs_grad = jnp.max(jnp.stack([jnp.max(jnp.abs(g)) if g.size > 0 else jnp.array(0.0) 
                                      for g in flat_grads]))
    
    # Compute min absolute non-zero gradient (useful for detecting underflow)
    def min_abs_nonzero(arr):
        abs_arr = jnp.abs(arr)
        abs_arr_nonzero = jnp.where(abs_arr > 0, abs_arr, jnp.full_like(abs_arr, jnp.inf))
        min_val = jnp.min(abs_arr_nonzero)
        return jnp.where(jnp.isinf(min_val), jnp.array(0.0), min_val)
    
    mins = jnp.stack([min_abs_nonzero(g) if g.size > 0 else jnp.array(jnp.inf) 
                      for g in flat_grads])
    min_abs_grad_nonzero = jnp.min(mins)
    min_abs_grad_nonzero = jnp.where(jnp.isinf(min_abs_grad_nonzero), 
                                     jnp.array(0.0), 
                                     min_abs_grad_nonzero)
    
    # Check for non-finite values
    has_nan = any(jnp.any(jnp.isnan(g)) for g in flat_grads)
    has_inf = any(jnp.any(jnp.isinf(g)) for g in flat_grads)
    
    # Compute ratio of finite values
    total_elements = sum(g.size for g in flat_grads)
    finite_elements = sum(jnp.sum(jnp.isfinite(g)) for g in flat_grads)
    finite_ratio = finite_elements / total_elements if total_elements > 0 else jnp.array(1.0)
    
    return {
        "grad_norm": grad_norm,
        "max_abs_grad": max_abs_grad,
        "min_abs_grad_nonzero": min_abs_grad_nonzero,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "finite_ratio": finite_ratio,
    }


def get_loss_scale_metrics(loss_scale: LossScale) -> Dict[str, jnp.ndarray]:
    """Get monitoring metrics for a loss scale.
    
    Args:
        loss_scale: Loss scale object.
        
    Returns:
        Dictionary with metrics about the loss scale.
    """
    metrics = {
        "loss_scale_value": loss_scale.loss_scale,
    }
    
    # Add specific metrics for dynamic loss scale
    if isinstance(loss_scale, DynamicLossScale):
        metrics.update({
            "loss_scale_growth_interval": jnp.array(loss_scale._growth_interval),
            "loss_scale_growth_factor": jnp.array(loss_scale._growth_factor),
            "loss_scale_backoff_factor": jnp.array(loss_scale._backoff_factor),
            "loss_scale_steps": jnp.array(loss_scale._steps_since_finite),
        })
    
    return metrics


def should_warn_about_precision(
    grads_stats: Dict[str, jnp.ndarray],
    loss_scale_value: jnp.ndarray,
    dtype: DType
) -> Tuple[bool, str]:
    """Check if there are potential precision issues based on gradient statistics.
    
    Args:
        grads_stats: Gradient statistics from compute_gradient_stats.
        loss_scale_value: Current loss scale value.
        dtype: Compute data type being used.
        
    Returns:
        Tuple of (should_warn, warning_message)
    """
    if grads_stats["has_nan"] or grads_stats["has_inf"]:
        return True, "Non-finite values detected in gradients."
    
    # For float16, check if loss scale might be too low/high
    if dtype == jnp.float16:
        max_representable = 65504.0
        
        # Check if we might be close to overflow
        if grads_stats["max_abs_grad"] * loss_scale_value > max_representable * 0.5:
            return True, (
                f"Loss scale ({loss_scale_value}) might be too high. "
                f"Max gradient ({grads_stats['max_abs_grad']}) * loss scale "
                f"is approaching float16 limit."
            )
            
        # Check if we might have underflow
        if grads_stats["min_abs_grad_nonzero"] < 1e-7 and grads_stats["grad_norm"] > 1e-5:
            return True, (
                f"Very small gradient values detected ({grads_stats['min_abs_grad_nonzero']}). "
                f"Consider increasing loss scale ({loss_scale_value})."
            )
    
    return False, "" 