"""Mixed precision utilities for XAX."""

import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util

# Type for a half-precision dtype, either float16 or bfloat16
HalfPrecisionDType = TypeVar("HalfPrecisionDType", jnp.float16, jnp.bfloat16)


def get_default_half_dtype() -> HalfPrecisionDType:
    """Get the default half-precision dtype for the current platform.
    
    Returns float16 for GPU and bfloat16 for TPU.
    Falls back to float16 on CPU.
    
    Returns:
        The default half-precision dtype.
    """
    platform = jax.default_backend()
    if platform == "tpu":
        return jnp.bfloat16
    else:
        return jnp.float16


@dataclass
class Policy:
    """A policy for mixed precision training.
    
    This class specifies the data types to use for parameters, computation,
    and output in a mixed precision training setup.
    
    Attributes:
        param_dtype: The data type for parameters
        compute_dtype: The data type for computation
        output_dtype: The data type for outputs
    """
    
    param_dtype: jnp.dtype
    compute_dtype: jnp.dtype
    output_dtype: jnp.dtype
    
    def __post_init__(self):
        """Validate the policy."""
        if not isinstance(self.param_dtype, jnp.dtype):
            self.param_dtype = jnp.dtype(self.param_dtype)
        if not isinstance(self.compute_dtype, jnp.dtype):
            self.compute_dtype = jnp.dtype(self.compute_dtype)
        if not isinstance(self.output_dtype, jnp.dtype):
            self.output_dtype = jnp.dtype(self.output_dtype)
    
    def cast_to_param(self, x: Any) -> Any:
        """Cast a value to the parameter data type.
        
        Args:
            x: The value to cast
            
        Returns:
            The cast value
        """
        return tree_map_dtype(x, self.param_dtype)
    
    def cast_to_compute(self, x: Any) -> Any:
        """Cast a value to the computation data type.
        
        Args:
            x: The value to cast
            
        Returns:
            The cast value
        """
        return tree_map_dtype(x, self.compute_dtype)
    
    def cast_to_output(self, x: Any) -> Any:
        """Cast a value to the output data type.
        
        Args:
            x: The value to cast
            
        Returns:
            The cast value
        """
        return tree_map_dtype(x, self.output_dtype)
    
    def with_param_dtype(self, dtype: jnp.dtype) -> "Policy":
        """Create a new policy with a different parameter data type.
        
        Args:
            dtype: The new parameter data type
            
        Returns:
            A new policy with the specified parameter data type
        """
        return Policy(dtype, self.compute_dtype, self.output_dtype)
    
    def with_compute_dtype(self, dtype: jnp.dtype) -> "Policy":
        """Create a new policy with a different computation data type.
        
        Args:
            dtype: The new computation data type
            
        Returns:
            A new policy with the specified computation data type
        """
        return Policy(self.param_dtype, dtype, self.output_dtype)
    
    def with_output_dtype(self, dtype: jnp.dtype) -> "Policy":
        """Create a new policy with a different output data type.
        
        Args:
            dtype: The new output data type
            
        Returns:
            A new policy with the specified output data type
        """
        return Policy(self.param_dtype, self.compute_dtype, dtype)


def get_policy(policy_str: str) -> Policy:
    """Create a policy from a string description.
    
    Args:
        policy_str: A string specifying the policy. Can be one of:
            - "default": Use float32 for everything
            - "float16": Use float16 for everything
            - "bfloat16": Use bfloat16 for everything
            - "half": Use the default half-precision dtype for everything
            - "mixed": Use float32 for parameters and output, half-precision for computation
            - A custom string like "params=float32,compute=float16,output=float32"
    
    Returns:
        The corresponding policy
        
    Raises:
        ValueError: If the policy string is invalid
    """
    full = jnp.float32
    half = get_default_half_dtype()
    
    if policy_str == "default":
        return Policy(full, full, full)
    elif policy_str == "float16":
        return Policy(jnp.float16, jnp.float16, jnp.float16)
    elif policy_str == "bfloat16":
        return Policy(jnp.bfloat16, jnp.bfloat16, jnp.bfloat16)
    elif policy_str == "half":
        return Policy(half, half, half)
    elif policy_str == "mixed":
        return Policy(full, half, full)
    
    # Parse a custom policy string
    try:
        parts = policy_str.split(",")
        param_dtype = full
        compute_dtype = full
        output_dtype = full
        
        for part in parts:
            key, value = part.split("=")
            if key == "params":
                param_dtype = _parse_dtype(value)
            elif key == "compute":
                compute_dtype = _parse_dtype(value)
            elif key == "output":
                output_dtype = _parse_dtype(value)
            else:
                raise ValueError(f"Unknown policy key: {key}")
        
        return Policy(param_dtype, compute_dtype, output_dtype)
    except Exception as e:
        raise ValueError(f"Invalid policy string: {policy_str}") from e


def _parse_dtype(dtype_str: str) -> jnp.dtype:
    """Parse a dtype string.
    
    Args:
        dtype_str: A string specifying a dtype
    
    Returns:
        The corresponding dtype
        
    Raises:
        ValueError: If the dtype string is invalid
    """
    if dtype_str == "float32":
        return jnp.float32
    elif dtype_str == "float16":
        return jnp.float16
    elif dtype_str == "bfloat16":
        return jnp.bfloat16
    elif dtype_str == "half":
        return get_default_half_dtype()
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")


def tree_map_dtype(pytree: Any, dtype: jnp.dtype) -> Any:
    """Apply dtype conversion to all arrays in a pytree.
    
    Args:
        pytree: A pytree of arrays or scalars
        dtype: The target data type
    
    Returns:
        A new pytree with all arrays converted to the target data type
    """
    def _cast(x):
        if hasattr(x, "dtype") and hasattr(x, "astype"):
            return x.astype(dtype)
        return x
    
    return jax.tree_util.tree_map(_cast, pytree)


class LossScale:
    """Base class for loss scaling strategies."""
    
    @property
    def loss_scale(self) -> jnp.ndarray:
        """Get the current loss scale value."""
        raise NotImplementedError("Subclasses must implement loss_scale")
    
    def scale(self, loss: jnp.ndarray) -> jnp.ndarray:
        """Scale the loss.
        
        Args:
            loss: The loss to scale
            
        Returns:
            The scaled loss
        """
        raise NotImplementedError("Subclasses must implement scale")
    
    def unscale(self, grads: Any) -> Any:
        """Unscale the gradients.
        
        Args:
            grads: The gradients to unscale
            
        Returns:
            The unscaled gradients
        """
        raise NotImplementedError("Subclasses must implement unscale")
    
    def adjust(self, grads_finite: jnp.ndarray) -> "LossScale":
        """Adjust the loss scale based on whether gradients are finite.
        
        Args:
            grads_finite: A boolean indicating whether all gradients are finite
            
        Returns:
            The updated loss scale
        """
        raise NotImplementedError("Subclasses must implement adjust")


class NoOpLossScale(LossScale):
    """A loss scale that does nothing.
    
    This is used when loss scaling is disabled.
    """
    
    @property
    def loss_scale(self) -> jnp.ndarray:
        """Get the current loss scale value."""
        return jnp.array(1.0, dtype=jnp.float32)
    
    def scale(self, loss: jnp.ndarray) -> jnp.ndarray:
        """Scale the loss (no-op).
        
        Args:
            loss: The loss to scale
            
        Returns:
            The unmodified loss
        """
        return loss
    
    def unscale(self, grads: Any) -> Any:
        """Unscale the gradients (no-op).
        
        Args:
            grads: The gradients to unscale
            
        Returns:
            The unmodified gradients
        """
        return grads
    
    def adjust(self, grads_finite: jnp.ndarray) -> "NoOpLossScale":
        """Adjust the loss scale (no-op).
        
        Args:
            grads_finite: A boolean indicating whether all gradients are finite
            
        Returns:
            self
        """
        return self


class StaticLossScale(LossScale):
    """A loss scale with a fixed value.
    
    This is a simple loss scaling strategy that uses a fixed scaling factor.
    
    Attributes:
        scale_value: The fixed scaling factor
    """
    
    def __init__(self, scale_value: float):
        """Initialize the loss scale.
        
        Args:
            scale_value: The fixed scaling factor
        """
        self._scale_value = jnp.array(scale_value, dtype=jnp.float32)
    
    @property
    def loss_scale(self) -> jnp.ndarray:
        """Get the current loss scale value."""
        return self._scale_value
    
    def scale(self, loss: jnp.ndarray) -> jnp.ndarray:
        """Scale the loss.
        
        Args:
            loss: The loss to scale
            
        Returns:
            The scaled loss
        """
        return loss * self._scale_value
    
    def unscale(self, grads: Any) -> Any:
        """Unscale the gradients.
        
        Args:
            grads: The gradients to unscale
            
        Returns:
            The unscaled gradients
        """
        def _unscale(g):
            return g / self._scale_value
        
        return jax.tree_util.tree_map(_unscale, grads)
    
    def adjust(self, grads_finite: jnp.ndarray) -> "StaticLossScale":
        """Adjust the loss scale (no-op for static loss scale).
        
        Args:
            grads_finite: A boolean indicating whether all gradients are finite
            
        Returns:
            self
        """
        return self


class DynamicLossScale(LossScale):
    """A loss scale that dynamically adjusts its value.
    
    This loss scaling strategy increases the scaling factor when gradients
    are consistently finite, and decreases it when gradients contain infinities
    or NaNs.
    
    Attributes:
        initial_scale: The initial scaling factor
        growth_interval: Number of consecutive finite gradient steps before increasing the scale
        growth_factor: Factor by which to increase the scale
        backoff_factor: Factor by which to decrease the scale
        max_scale: Maximum allowed scale value (or None for no limit)
        counter: Counter for consecutive finite gradient steps
    """
    
    def __init__(
        self,
        initial_scale: float = 2**15,
        growth_interval: int = 2000,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        max_scale: Optional[float] = None,
        counter: Optional[int] = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize the loss scale.
        
        Args:
            initial_scale: The initial scaling factor
            growth_interval: Number of consecutive finite gradient steps before increasing the scale
            growth_factor: Factor by which to increase the scale
            backoff_factor: Factor by which to decrease the scale
            max_scale: Maximum allowed scale value (or None for no limit)
            counter: Counter for consecutive finite gradient steps
        """
        self._scale_value = jnp.array(initial_scale, dtype=dtype)
        self._growth_interval = growth_interval
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._max_scale = jnp.array(max_scale, dtype=dtype) if max_scale is not None else None
        self._counter = jnp.array(0 if counter is None else counter, dtype=jnp.int32)
    
    @property
    def loss_scale(self) -> jnp.ndarray:
        """Get the current loss scale value."""
        return self._scale_value
    
    def scale(self, loss: jnp.ndarray) -> jnp.ndarray:
        """Scale the loss.
        
        Args:
            loss: The loss to scale
            
        Returns:
            The scaled loss
        """
        return loss * self._scale_value
    
    def unscale(self, grads: Any) -> Any:
        """Unscale the gradients.
        
        Args:
            grads: The gradients to unscale
            
        Returns:
            The unscaled gradients
        """
        def _unscale(g):
            return g / self._scale_value
        
        return jax.tree_util.tree_map(_unscale, grads)
    
    def adjust(self, grads_finite: jnp.ndarray) -> "DynamicLossScale":
        """Adjust the loss scale based on whether gradients are finite.
        
        If gradients are finite, increment the counter. If the counter reaches
        the growth interval, increase the scale and reset the counter.
        
        If gradients are not finite, decrease the scale and reset the counter.
        
        Args:
            grads_finite: A boolean indicating whether all gradients are finite
            
        Returns:
            The updated loss scale
        """
        if hasattr(grads_finite, "item") and callable(grads_finite.item):
            grads_finite = bool(grads_finite.item())
        
        if not grads_finite:
            # Decrease the scale and reset the counter
            new_scale = self._scale_value * self._backoff_factor
            new_counter = jnp.array(0, dtype=jnp.int32)
        else:
            # Increment the counter
            new_counter = self._counter + 1
            
            if new_counter >= self._growth_interval:
                # Increase the scale and reset the counter
                new_scale = self._scale_value * self._growth_factor
                if self._max_scale is not None:
                    new_scale = jnp.minimum(new_scale, self._max_scale)
                new_counter = jnp.array(0, dtype=jnp.int32)
            else:
                # Keep the same scale
                new_scale = self._scale_value
        
        return DynamicLossScale(
            initial_scale=new_scale,
            growth_interval=self._growth_interval,
            growth_factor=self._growth_factor,
            backoff_factor=self._backoff_factor,
            max_scale=self._max_scale.item() if self._max_scale is not None else None,
            counter=new_counter.item(),
            dtype=self._scale_value.dtype,
        )


def all_finite(tree: Any) -> jnp.ndarray:
    """Check if all values in a pytree are finite.
    
    Args:
        tree: A pytree of arrays or scalars
    
    Returns:
        A boolean indicating whether all values are finite
    """
    def _check_finite(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.inexact):
            return jnp.all(jnp.isfinite(x))
        return jnp.array(True)
    
    leaves = jax.tree_util.tree_map(_check_finite, tree)
    leaves_flat = jax.tree_util.tree_leaves(leaves)
    return jnp.all(jnp.stack(leaves_flat))


def select_tree(pred: jnp.ndarray, on_true: Any, on_false: Any) -> Any:
    """Select between two pytrees based on a predicate.
    
    Args:
        pred: A boolean predicate
        on_true: The pytree to select if pred is True
        on_false: The pytree to select if pred is False
    
    Returns:
        Either on_true or on_false, based on pred
    """
    return jax.tree_util.tree_map(
        lambda t, f: jnp.where(pred, t, f),
        on_true,
        on_false
    ) 