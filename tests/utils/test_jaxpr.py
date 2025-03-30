"""Tests for the JAXPR utilities."""

from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Array

import xax


def test_save_jaxpr_dot(tmpdir: Path) -> None:
    def loss_fn(x: Array) -> Array:
        return jnp.sum(x**2)

    jaxpr = jax.make_jaxpr(loss_fn)(jnp.array([1.0, 2.0, 3.0]))
    filename = tmpdir / "test_jaxpr.dot"
    xax.save_jaxpr_dot(jaxpr, filename)

    with open(filename, "r") as f:
        contents = f.read()

    # Ensures that the .dot file is formatted correctly (this is not a very
    # comprehensive test, we should probably improve it).)
    assert contents.startswith("digraph Jaxpr {")
    assert contents.endswith("}\n")
