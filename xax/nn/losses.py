"""Defines some common loss functions."""

import jax.numpy as jnp
from jaxtyping import Array


def cross_entropy(y: Array, pred_y: Array, axis: int = 1) -> Array:
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, axis), axis=axis)
    return -jnp.mean(pred_y)
