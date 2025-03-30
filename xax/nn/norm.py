"""Normalization utilities."""

from typing import Literal, cast, get_args

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
