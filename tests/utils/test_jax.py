"""Tests Jax functions."""

import jax.numpy as jnp
from jaxtyping import Array

import xax


def test_scan_accumulate() -> None:
    def scan_fn(carry: Array, x: Array) -> tuple[Array, Array]:
        return carry + x, x

    init = jnp.array(0)
    xs = jnp.arange(10)
    carry, ys = xax.scan(scan_fn, init, xs, jit_level=-1)
    assert jnp.allclose(carry, jnp.sum(xs))
    assert jnp.allclose(ys, xs)


def test_scan_with_length() -> None:
    def scan_fn(carry: Array, x: Array) -> tuple[Array, Array]:
        return carry + x, x

    init = jnp.array(0)
    xs = jnp.arange(10)
    carry, ys = xax.scan(scan_fn, init, xs, length=10, jit_level=-1)
    assert jnp.allclose(carry, jnp.sum(xs))
    assert jnp.allclose(ys, xs)


def test_scan_with_pytree_carry() -> None:
    def scan_fn(carry: tuple[Array, Array], x: Array) -> tuple[tuple[Array, Array], Array]:
        return (carry[0] + x, carry[1]), x

    init = (jnp.array(0), jnp.array(0))
    xs = jnp.arange(10)
    carry, ys = xax.scan(scan_fn, init, xs, length=10, jit_level=-1)
    assert jnp.allclose(carry[0], jnp.sum(xs))
    assert jnp.allclose(carry[1], jnp.zeros_like(carry[1]))
    assert jnp.allclose(ys, xs)
