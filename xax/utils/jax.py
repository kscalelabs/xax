"""Defines some utility functions for interfacing with Jax."""

import jax.numpy as jnp
import numpy as np

Number = int | float | np.ndarray | jnp.ndarray


def as_float(value: int | float | np.ndarray | jnp.ndarray) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return float(value.item())
    raise TypeError(f"Unexpected type: {type(value)}")
