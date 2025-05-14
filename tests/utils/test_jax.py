"""Tests Jax functions."""

import jax
import jax.numpy as jnp
from jaxtyping import Array

import xax


def test_scan_accumulate() -> None:
    def scan_fn(carry: Array, x: Array) -> tuple[Array, Array]:
        return carry + x, x

    init = jnp.array(0)
    xs = jnp.arange(10)
    expected_carry, expected_ys = jax.lax.scan(scan_fn, init, xs)
    carry, ys = xax.scan(scan_fn, init, xs, jit_level=-1)
    assert jnp.allclose(carry, expected_carry)
    assert jnp.allclose(ys, expected_ys)


def test_scan_with_length() -> None:
    def scan_fn(carry: Array, x: Array) -> tuple[Array, Array]:
        return carry + x, x

    init = jnp.array(0)
    xs = jnp.arange(10)
    expected_carry, expected_ys = jax.lax.scan(scan_fn, init, xs, length=10)
    carry, ys = xax.scan(scan_fn, init, xs, length=10, jit_level=-1)
    assert jnp.allclose(carry, expected_carry)
    assert jnp.allclose(ys, expected_ys)


def test_scan_with_pytree_carry() -> None:
    def scan_fn(carry: tuple[Array, Array], x: Array) -> tuple[tuple[Array, Array], Array]:
        return (carry[0] + x, carry[1]), x

    init = (jnp.array(0), jnp.array(0))
    xs = jnp.arange(10)
    carry, ys = xax.scan(scan_fn, init, xs, length=10, jit_level=-1)
    expected_carry, expected_ys = jax.lax.scan(scan_fn, init, xs, length=10)
    assert jnp.allclose(carry[0], expected_carry[0])
    assert jnp.allclose(carry[1], expected_carry[1])
    assert jnp.allclose(ys, expected_ys)


def test_scan_reverse() -> None:
    def scan_fn(carry: Array, x: Array) -> tuple[Array, Array]:
        return carry + x, carry

    init = jnp.array(0)
    xs = jnp.arange(5)
    expected_carry, expected_ys = jax.lax.scan(scan_fn, init, xs, reverse=True)
    carry, ys = xax.scan(scan_fn, init, xs, reverse=True, jit_level=-1)
    assert jnp.allclose(carry, expected_carry)
    assert jnp.allclose(ys, expected_ys)


def test_scan_without_xs() -> None:
    def scan_fn(carry: Array, x: None) -> tuple[Array, Array]:
        return carry + 1, carry

    init = jnp.array(0)
    carry, ys = xax.scan(scan_fn, init, xs=None, length=5, jit_level=-1)
    expected_ys = jnp.arange(5)
    assert jnp.allclose(carry, jnp.array(5))
    assert jnp.allclose(ys, expected_ys)


def test_scan_empty() -> None:
    def scan_fn(carry: Array, x: Array) -> tuple[Array, Array]:
        return carry + x, x

    init = jnp.array(0)
    xs = jnp.array([])
    expected_carry, expected_ys = jax.lax.scan(scan_fn, init, xs)
    carry, ys = xax.scan(scan_fn, init, xs, jit_level=-1)
    assert jnp.allclose(carry, expected_carry)
    assert jnp.allclose(ys, expected_ys)


def test_scan_with_jit() -> None:
    def scan_fn(carry: Array, x: Array) -> tuple[Array, Array]:
        return carry + x, x

    init = jnp.array(0)
    xs = jnp.arange(10)

    # Test with JIT enabled
    carry_jit, ys_jit = xax.scan(scan_fn, init, xs, jit_level=None)

    # Test with JIT disabled
    carry_no_jit, ys_no_jit = xax.scan(scan_fn, init, xs, jit_level=-1)

    assert jnp.allclose(carry_jit, carry_no_jit)
    assert jnp.allclose(ys_jit, ys_no_jit)


def test_vmap_basic() -> None:
    def fn(x: Array) -> Array:
        return x * 2

    xs = jnp.arange(10)
    result = xax.vmap(fn, jit_level=-1)(xs)
    expected = xs * 2
    assert jnp.allclose(result, expected)


def test_vmap_multiple_args() -> None:
    def fn(x: Array, y: Array) -> Array:
        return x + y

    xs = jnp.arange(5)
    ys = jnp.arange(5, 10)
    result = xax.vmap(fn, in_axes=(0, 0), jit_level=-1)(xs, ys)
    expected = xs + ys
    assert jnp.allclose(result, expected)


def test_vmap_different_axes() -> None:
    def fn(x: Array, y: Array) -> Array:
        return x + y

    # x has shape (3, 4), y has shape (4, 3)
    x = jnp.ones((3, 4))
    y = jnp.ones((4, 3))

    # Map over axis 0 of x and axis 1 of y
    result = xax.vmap(fn, in_axes=(0, 1), jit_level=-1)(x, y)
    expected = jnp.ones((3, 4)) + jnp.ones((3, 4))
    assert result.shape == (3, 4)
    assert jnp.allclose(result, expected)


def test_vmap_complex_function() -> None:
    def fn(x: Array) -> tuple[Array, Array]:
        return x * 2, x**2

    xs = jnp.arange(5)
    result1, result2 = xax.vmap(fn, jit_level=-1)(xs)
    expected1 = xs * 2
    expected2 = xs**2
    assert jnp.allclose(result1, expected1)
    assert jnp.allclose(result2, expected2)


def test_vmap_pytree_input() -> None:
    def fn(tree: tuple[Array, Array]) -> Array:
        x, y = tree
        return x + y

    xs = (jnp.arange(5), jnp.arange(5, 10))
    result = xax.vmap(fn, jit_level=-1)(xs)
    expected = jnp.arange(5) + jnp.arange(5, 10)
    assert jnp.allclose(result, expected)


def test_vmap_with_jit() -> None:
    def fn(x: Array) -> Array:
        return x * 3

    xs = jnp.arange(10)

    # Test with JIT enabled
    result_jit = xax.vmap(fn, jit_level=None)(xs)

    # Test with JIT disabled
    result_no_jit = xax.vmap(fn, jit_level=-1)(xs)

    assert jnp.allclose(result_jit, result_no_jit)


def test_vmap_empty_input() -> None:
    def fn(x: Array) -> Array:
        return x * 2

    xs = jnp.array([])
    result = xax.vmap(fn, jit_level=-1)(xs)
    assert result.shape == (0,)
