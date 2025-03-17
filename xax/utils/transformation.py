"""Transformations that work with Equinox, NNX, etc. with JAX-like API."""

from typing import Any, Callable

import equinox as eqx
import jax
from jaxtyping import PyTree


def scan_model(
    fn: Callable[[tuple[PyTree, ...], PyTree], tuple[tuple[PyTree, ...], PyTree]],
    stateful_model_init: PyTree,
    rest_init: tuple[PyTree, ...],
    xs: PyTree | None = None,
    length: int | None = None,
    mutable_criterion: Callable[[Any], bool] = eqx.is_inexact_array,
) -> tuple[tuple[PyTree, ...], PyTree]:
    """Scan that works with models that have both mutable and static parts.

    This is useful for training models that have both mutable and static
    parts, such as RNNs. This lets you achieve similar behavior with Equinox
    models as you would with the Flax linen API.

    Args:
        fn: The function to scan.
        stateful_model_init: The initial stateful model.
        rest_init: The initial rest of the model.
        xs: The input to the scan.
        length: The length of the scan.
        mutable_criterion: The criterion for determining whether a part of
            the model is mutable.

    Returns:
        The output of the scan.
    """
    # partitioning separates a model into effectively params and functions
    # inexact is the criterion that JAX uses for this
    mutable_model, static_model = eqx.partition(stateful_model_init, mutable_criterion)

    def wrapped_fn(carry: tuple[PyTree, ...], x: PyTree) -> tuple[tuple[PyTree, ...], PyTree]:
        mutable_model, *other_state = carry
        full_model = eqx.combine(mutable_model, static_model)
        new_carry, y = fn((full_model, *other_state), x)
        new_model, *new_other = new_carry
        new_mutable_model, _ = eqx.partition(new_model, mutable_criterion)
        return (new_mutable_model, *new_other), y

    init_carry = (mutable_model, *rest_init)
    final_carry, outputs = jax.lax.scan(wrapped_fn, init_carry, xs, length=length)
    final_mutable_model, *final_rest = final_carry

    final_model = eqx.combine(final_mutable_model, static_model)
    return (final_model, *final_rest), outputs
