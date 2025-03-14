"""Defines geometry functions."""

import jax
from jax import numpy as jnp


def quat_to_euler(quat_4: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Normalizes and converts a quaternion (w, x, y, z) to roll, pitch, yaw.

    Args:
        quat_4: The quaternion to convert, shape (*, 4).
        eps: A small epsilon value to avoid division by zero.

    Returns:
        The roll, pitch, yaw angles with shape (*, 3).
    """
    quat_4 = quat_4 / (jnp.linalg.norm(quat_4, axis=-1, keepdims=True) + eps)
    w, x, y, z = jnp.split(quat_4, 4, axis=-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)

    # Handle edge cases where |sinp| >= 1
    pitch = jnp.where(
        jnp.abs(sinp) >= 1.0,
        jnp.sign(sinp) * jnp.pi / 2.0,  # Use 90 degrees if out of range
        jnp.arcsin(sinp),
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return jnp.concatenate([roll, pitch, yaw], axis=-1)


def euler_to_quat(euler_3: jax.Array) -> jax.Array:
    """Converts roll, pitch, yaw angles to a quaternion (w, x, y, z).

    Args:
        euler_3: The roll, pitch, yaw angles, shape (*, 3).

    Returns:
        The quaternion with shape (*, 4).
    """
    # Extract roll, pitch, yaw from input
    roll, pitch, yaw = jnp.split(euler_3, 3, axis=-1)

    # Calculate trigonometric functions for each angle
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    # Calculate quaternion components using the conversion formula
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # Combine into quaternion [w, x, y, z]
    quat = jnp.concatenate([w, x, y, z], axis=-1)

    # Normalize the quaternion
    quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)

    return quat


def get_projected_gravity_vector_from_quat(quat: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Calculates the gravity vector projected onto the local frame given a quaternion orientation.

    Args:
        quat: A quaternion (w,x,y,z) representing the orientation, shape (*, 4).
        eps: A small epsilon value to avoid division by zero.

    Returns:
        A 3D vector representing the gravity in the local frame, shape (*, 3).
    """
    # Normalize quaternion
    quat = quat / (jnp.linalg.norm(quat, axis=-1, keepdims=True) + eps)
    w, x, y, z = jnp.split(quat, 4, axis=-1)

    # Gravity vector in world frame is [0, 0, -1] (pointing down)
    # Rotate gravity vector using quaternion rotation

    # Calculate quaternion rotation: q * [0,0,-1] * q^-1
    gx = 2 * (x * z - w * y)
    gy = 2 * (y * z + w * x)
    gz = w * w - x * x - y * y + z * z

    # Note: We're rotating [0,0,-1], so we negate gz to match the expected direction
    return jnp.concatenate([gx, gy, -gz], axis=-1)
