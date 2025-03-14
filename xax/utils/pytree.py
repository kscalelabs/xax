"""Utils for accessing, modifying, and otherwise manipulating pytrees."""

from typing import Any

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree


def slice_array(x: Array, start: Array, slice_length: int) -> Array:
    """Get a slice of an array along the first dimension.

    For multi-dimensional arrays, this slices only along the first dimension
    and keeps all other dimensions intact.
    """
    chex.assert_shape(start, ())
    chex.assert_shape(slice_length, ())
    start_indices = (start,) + (0,) * (len(x.shape) - 1)
    slice_sizes = (slice_length,) + x.shape[1:]

    return jax.lax.dynamic_slice(x, start_indices, slice_sizes)


def slice_pytree(pytree: PyTree, start: Array, slice_length: int) -> PyTree:
    """Get a slice of a pytree."""
    return jax.tree_util.tree_map(lambda x: slice_array(x, start, slice_length), pytree)


def flatten_array(x: Array, flatten_size: int) -> Array:
    """Flatten an array into a (flatten_size, ...) array."""
    reshaped = jnp.reshape(x, (flatten_size, *x.shape[2:]))
    assert reshaped.shape[0] == flatten_size
    return reshaped


def flatten_pytree(pytree: PyTree, flatten_size: int) -> PyTree:
    """Flatten a pytree into a (flatten_size, ...) pytree."""
    return jax.tree_util.tree_map(lambda x: flatten_array(x, flatten_size), pytree)


def pytree_has_nans(pytree: PyTree) -> Array:
    """Check if a pytree has any NaNs."""
    has_nans = jax.tree_util.tree_reduce(
        lambda a, b: jnp.logical_or(a, b),
        jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), pytree),
    )
    return has_nans


def update_pytree(cond: Array, new: PyTree, original: PyTree) -> PyTree:
    """Update a pytree based on a condition."""
    # Tricky, need use tree_map because where expects array leafs.
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), new, original)


def compute_nan_ratio(pytree: PyTree) -> Array:
    """Computes the ratio of NaNs vs non-NaNs in a given PyTree."""
    nan_counts = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.isnan(x)), pytree)
    total_counts = jax.tree_util.tree_map(lambda x: x.size, pytree)

    total_nans = jax.tree_util.tree_reduce(lambda a, b: a + b, nan_counts, 0)
    total_elements = jax.tree_util.tree_reduce(lambda a, b: a + b, total_counts, 0)
    overall_nan_ratio = jnp.array(total_nans / total_elements)

    return overall_nan_ratio


def reshuffle_pytree(data: PyTree, batch_shape: tuple[int, ...], rng: PRNGKeyArray) -> PyTree:
    """Reshuffle a rollout array across arbitrary batch dimensions."""
    rngs = jax.random.split(rng, len(batch_shape))
    perms = [jax.random.choice(rng_i, jnp.arange(dim), (dim,)) for rng_i, dim in zip(rngs, batch_shape)]

    # n-dimensional index grid from permutations
    idx_grids = jnp.meshgrid(*perms, indexing="ij")

    def permute_array(x: Any) -> Array:  # noqa: ANN401
        if isinstance(x, Array):
            return x[tuple(idx_grids)]
        return x

    return jax.tree_util.tree_map(permute_array, data)
