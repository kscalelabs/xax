"""Norm and metric utilities."""

from typing import Literal, cast, get_args

import jax
import jax.numpy as jnp
from jaxtyping import Array

NormType = Literal["l1", "l2"]


def cast_norm_type(norm: str) -> NormType:
    if norm not in get_args(NormType):
        raise ValueError(f"Invalid norm: {norm}")
    return cast(NormType, norm)


def get_norm(x: Array, norm: NormType) -> Array:
    match norm:
        case "l1":
            return jnp.abs(x)
        case "l2":
            return jnp.square(x)
        case _:
            raise ValueError(f"Invalid norm: {norm}")


def compute_distance_matrix(a: Array, b: Array) -> Array:
    has_features = len(a.shape) > 1
    a = jnp.expand_dims(a, axis=1)
    b = jnp.expand_dims(b, axis=0)
    distance_matrix = jnp.square(a - b)
    if has_features:
        distance_matrix = jnp.sum(distance_matrix, axis=-1)
    return distance_matrix


def pad_inf(inp: Array, before: int, after: int) -> Array:
    return jnp.pad(inp, (before, after), constant_values=jnp.inf)


def dtw(prediction: Array, target: Array) -> Array:
    """Dynamic Time Warping.

    Reference:
        K. Heidler. (Soft-)DTW for JAX, Github, https://github.com/khdlr/softdtw_jax
    """
    distance_matrix = compute_distance_matrix(prediction, target)
    # contract: height >= width
    if distance_matrix.shape[0] < distance_matrix.shape[1]:
        distance_matrix = distance_matrix.T
    height, width = distance_matrix.shape

    rows = []
    for row in range(height):
        rows.append(pad_inf(distance_matrix[row], row, height - row - 1))

    model_matrix = jnp.stack(rows, axis=1)

    init = (pad_inf(model_matrix[0], 1, 0), pad_inf(model_matrix[1] + model_matrix[0, 0], 1, 0))

    def _scan_step(carry: tuple[Array, Array], current_antidiagonal: Array) -> tuple[tuple[Array, Array], Array]:
        two_ago, one_ago = carry

        diagonal = two_ago[:-1]
        right = one_ago[:-1]
        down = one_ago[1:]
        best = jnp.min(jnp.stack([diagonal, right, down], axis=-1), axis=-1)

        next_row = best + current_antidiagonal
        next_row = pad_inf(next_row, 1, 0)

        return (one_ago, next_row), next_row

    carry, ys = jax.lax.scan(_scan_step, init, model_matrix[2:], unroll=4)
    return carry[1][-1]
