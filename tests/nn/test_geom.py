"""Tests geometry functions."""

import jax
import pytest
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
    """Test that Euler angles and quaternions represent the same rotation."""
    rng = jax.random.PRNGKey(0)
    euler = jax.random.uniform(rng, (10, 3), minval=-jnp.pi, maxval=jnp.pi)
    quat = xax.euler_to_quat(euler)
    euler_again = xax.quat_to_euler(quat)
    quat_again = xax.euler_to_quat(euler_again)

    # Quaternions q and -q represent the same rotation
    # We'll compare the absolute dot product which should be close to 1
    dot_product = jnp.abs(jnp.sum(quat * quat_again, axis=-1))
    assert jnp.allclose(dot_product, 1.0, atol=1e-5)


def test_quat_to_euler_special_cases() -> None:
    """Test quaternion to Euler conversion for special cases."""
    # Test gimbal lock case (pitch = 90 degrees)
    # This is approximately a rotation of 90 degrees around Y axis
    gimbal_lock_quat = jnp.array([0.7071, 0, 0.7071, 0])
    euler = xax.quat_to_euler(gimbal_lock_quat)

    # Pitch should be close to pi/2
    assert jnp.isclose(euler[1], jnp.pi / 2, atol=1e-2)

    # Test non-normalized quaternion
    non_normalized = jnp.array([2.0, 1.0, 0.5, 0.3])
    euler = xax.quat_to_euler(non_normalized)

    # Should still work and give valid Euler angles
    normalized_euler = xax.quat_to_euler(non_normalized / jnp.linalg.norm(non_normalized))
    assert jnp.allclose(euler, normalized_euler)


def test_euler_to_quat_batch() -> None:
    """Test batch processing of Euler angles to quaternions."""
    # Test with a batch of Euler angles
    batch_euler = jnp.array(
        [
            [0, 0, 0],  # Identity
            [jnp.pi / 4, 0, 0],  # 45 deg roll
            [0, jnp.pi / 4, 0],  # 45 deg pitch
            [0, 0, jnp.pi / 4],  # 45 deg yaw
        ]
    )

    batch_quat = xax.euler_to_quat(batch_euler)

    # Expected quaternions for these rotations
    expected_quat = jnp.array(
        [
            [1, 0, 0, 0],  # Identity
            [jnp.cos(jnp.pi / 8), jnp.sin(jnp.pi / 8), 0, 0],  # 45 deg roll
            [jnp.cos(jnp.pi / 8), 0, jnp.sin(jnp.pi / 8), 0],  # 45 deg pitch
            [jnp.cos(jnp.pi / 8), 0, 0, jnp.sin(jnp.pi / 8)],  # 45 deg yaw
        ]
    )

    assert jnp.allclose(batch_quat, expected_quat, atol=1e-5)


@pytest.mark.parametrize(
    "vector, euler, expected",
    [
        (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0])),
        (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, jnp.pi / 2]), jnp.array([0.0, 1.0, 0.0])),
        (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, jnp.pi / 2, 0.0]), jnp.array([0.0, 0.0, -1.0])),
        (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, jnp.pi / 2]), jnp.array([0.0, 1.0, 0.0])),
    ],
)
def test_rotate_vector_by_quat(
    vector: jax.Array,
    euler: jax.Array,
    expected: jax.Array,
) -> None:
    quat = xax.euler_to_quat(euler)
    rotated_vector = xax.rotate_vector_by_quat(vector, quat)
    assert jnp.allclose(rotated_vector, expected)


def test_get_projected_gravity_vector_from_quat() -> None:
    """Test gravity vector projection from quaternion orientation."""
    # Identity quaternion (no rotation) - gravity should point down in local frame
    identity_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    gravity = xax.get_projected_gravity_vector_from_quat(identity_quat)
    assert jnp.allclose(gravity, jnp.array([0.0, 0.0, -1.0]), atol=1e-3)

    # 90 degree rotation around X-axis - gravity should point along Y
    x90_quat = jnp.array([jnp.cos(jnp.pi / 4), jnp.sin(jnp.pi / 4), 0.0, 0.0])
    gravity = xax.get_projected_gravity_vector_from_quat(x90_quat)
    assert jnp.allclose(gravity, jnp.array([0.0, 1.0, 0.0]), atol=1e-3)

    # 90 degree rotation around Y-axis - gravity should point along X
    y90_quat = jnp.array([jnp.cos(jnp.pi / 4), 0.0, jnp.sin(jnp.pi / 4), 0.0])
    gravity = xax.get_projected_gravity_vector_from_quat(y90_quat)
    assert jnp.allclose(gravity, jnp.array([-1.0, 0.0, 0.0]), atol=1e-3)

    # Test with batch of quaternions
    batch_quat = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # Identity
            [jnp.cos(jnp.pi / 4), jnp.sin(jnp.pi / 4), 0.0, 0.0],  # 90 deg around X
            [jnp.cos(jnp.pi / 4), 0.0, jnp.sin(jnp.pi / 4), 0.0],  # 90 deg around Y
        ]
    )

    batch_gravity = jax.vmap(xax.get_projected_gravity_vector_from_quat)(batch_quat)
    expected_gravity = jnp.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert jnp.allclose(batch_gravity, expected_gravity, atol=1e-5)
