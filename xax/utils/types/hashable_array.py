"""Defines a hashable array wrapper.

Since Jax relies extensively on hashing, and we sometimes want to treat Jax
arrays as constants, this wrapper lets us ensure that Jax and Numpy arrays can
be hashed for Jitting.
"""

import jax.numpy as jnp
import numpy as np


class HashableArray:
    def __init__(self, array: np.ndarray | jnp.ndarray) -> None:
        if not isinstance(array, (np.ndarray, jnp.ndarray)):
            raise ValueError(f"Expected np.ndarray or jnp.ndarray, got {type(array)}")
        self.array = array
        self._hash: int | None = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.array.tobytes())
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashableArray):
            return False
        return bool(jnp.array_equal(self.array, other.array))


def hashable_array(array: np.ndarray | jnp.ndarray) -> HashableArray:
    return HashableArray(array)
