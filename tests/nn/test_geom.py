"""Tests geometry functions."""

import jax
from jax import numpy as jnp

import xax


def test_quat_to_euler() -> None:
    quat = jnp.array([1, 0, 0, 0])
    euler = xax.quat_to_euler(quat)
    assert jnp.allclose(euler, jnp.array([0, 0, 0]))


def test_euler_to_quat() -> None:
    euler = jnp.array([0, 0, 0])
    quat = xax.euler_to_quat(euler)
    assert jnp.allclose(quat, jnp.array([1, 0, 0, 0]))


def test_rotation_equivalence() -> None:
    rng = jax.random.PRNGKey(0)
    euler = jax.random.uniform(rng, (10, 3), minval=-jnp.pi, maxval=jnp.pi)
    quat = xax.euler_to_quat(euler)
    quat = xax.euler_to_quat(euler)
    euler_again = xax.quat_to_euler(quat)
    quat_again = xax.euler_to_quat(euler_again)
    dot_product = jnp.abs(jnp.sum(quat * quat_again, axis=-1))
    assert jnp.allclose(dot_product, 1.0, atol=1e-5)
