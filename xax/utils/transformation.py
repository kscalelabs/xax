"""Transformations that work with Equinox, NNX, etc. with JAX-like API."""

from typing import Callable

import equinox as eqx
import jax
from jaxtyping import PyTree


def scan_model(
    fn: Callable[[tuple[PyTree, ...], PyTree], tuple[tuple[PyTree, ...], PyTree]],
    stateful_model_init: PyTree,
    rest_init: tuple[PyTree, ...],
    xs: PyTree | None,
    length: int | None = None,
) -> tuple[tuple[PyTree, ...], PyTree]:
    """Scan that works with models that have both mutable and static parts."""
    # partitioning separates a model into effectively params and functions
    # inexact is the criterion that JAX uses for this
    mutable_model, static_model = eqx.partition(stateful_model_init, eqx.is_inexact_array)

    def wrapped_fn(carry: tuple[PyTree, ...], x: PyTree) -> tuple[tuple[PyTree, ...], PyTree]:
        mutable_model, *other_state = carry
        full_model = eqx.combine(mutable_model, static_model)
        new_carry, y = fn((full_model, *other_state), x)
        new_model, *new_other = new_carry
        new_mutable_model, _ = eqx.partition(new_model, eqx.is_inexact_array)
        return (new_mutable_model, *new_other), y

    init_carry = (mutable_model, *rest_init)
    final_carry, outputs = jax.lax.scan(wrapped_fn, init_carry, xs, length=length)
    final_mutable_model, *final_rest = final_carry

    final_model = eqx.combine(final_mutable_model, static_model)
    return (final_model, *final_rest), outputs
