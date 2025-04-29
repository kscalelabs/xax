"""Defines geometry functions."""

import chex
from jax import numpy as jnp
from jaxtyping import Array


def quat_to_euler(quat_4: Array, eps: float = 1e-6) -> Array:
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


def euler_to_quat(euler_3: Array) -> Array:
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


def get_projected_gravity_vector_from_quat(quat: Array, eps: float = 1e-6) -> Array:
    """Calculates the gravity vector projected onto the local frame given a quaternion orientation.

    Args:
        quat: A quaternion (w,x,y,z) representing the orientation, shape (*, 4).
        eps: A small epsilon value to avoid division by zero.

    Returns:
        A 3D vector representing the gravity in the local frame, shape (*, 3).
    """
    return rotate_vector_by_quat(jnp.array([0, 0, -9.81]), quat, inverse=True, eps=eps)


def rotate_vector_by_quat(vector: Array, quat: Array, inverse: bool = False, eps: float = 1e-6) -> Array:
    """Rotates a vector by a quaternion.

    Args:
        vector: The vector to rotate, shape (*, 3).
        quat: The quaternion to rotate by, shape (*, 4).
        inverse: If True, rotate the vector by the conjugate of the quaternion.
        eps: A small epsilon value to avoid division by zero.

    Returns:
        The rotated vector, shape (*, 3).
    """
    # Normalize quaternion
    quat = quat / (jnp.linalg.norm(quat, axis=-1, keepdims=True) + eps)
    w, x, y, z = jnp.split(quat, 4, axis=-1)

    if inverse:
        x, y, z = -x, -y, -z

    # Extract vector components
    vx, vy, vz = jnp.split(vector, 3, axis=-1)

    # Terms for x component
    xx = (
        w * w * vx
        + 2 * y * w * vz
        - 2 * z * w * vy
        + x * x * vx
        + 2 * y * x * vy
        + 2 * z * x * vz
        - z * z * vx
        - y * y * vx
    )

    # Terms for y component
    yy = (
        2 * x * y * vx
        + y * y * vy
        + 2 * z * y * vz
        + 2 * w * z * vx
        - z * z * vy
        + w * w * vy
        - 2 * w * x * vz
        - x * x * vy
    )

    # Terms for z component
    zz = (
        2 * x * z * vx
        + 2 * y * z * vy
        + z * z * vz
        - 2 * w * y * vx
        + w * w * vz
        + 2 * w * x * vy
        - y * y * vz
        - x * x * vz
    )

    return jnp.concatenate([xx, yy, zz], axis=-1)


def cubic_bezier_interpolation(y_start: Array, y_end: Array, x: Array) -> Array:
    """Cubic bezier interpolation.

    This is a cubic bezier curve that starts at y_start and ends at y_end,
    and is controlled by the parameter x. The curve is defined by the following formula:

    y(x) = y_start + (y_end - y_start) * (x**3 + 3 * (x**2 * (1 - x)))

    Args:
        y_start: The start value, shape (*).
        y_end: The end value, shape (*).
        x: The interpolation parameter, shape (*).

    Returns:
        The interpolated value, shape (*).
    """
    y_diff = y_end - y_start
    bezier = x**3 + 3 * (x**2 * (1 - x))
    return y_start + y_diff * bezier


def quat_to_rotmat(quat: Array, eps: float = 1e-6) -> Array:
    """Converts a quaternion to a rotation matrix.

    Args:
        quat: The quaternion to convert, shape (*, 4).
        eps: A small epsilon value to avoid division by zero.

    Returns:
        The rotation matrix, shape (*, 3, 3).
    """
    quat = quat / (jnp.linalg.norm(quat, axis=-1, keepdims=True) + eps)
    w, x, y, z = jnp.split(quat, 4, axis=-1)

    xx = 1 - 2 * (y * y + z * z)
    xy = 2 * (x * y - z * w)
    xz = 2 * (x * z + y * w)
    yx = 2 * (x * y + z * w)
    yy = 1 - 2 * (x * x + z * z)
    yz = 2 * (y * z - x * w)
    zx = 2 * (x * z - y * w)
    zy = 2 * (y * z + x * w)
    zz = 1 - 2 * (x * x + y * y)

    # Corrected stacking: row-major order
    return jnp.concatenate(
        [
            jnp.concatenate([xx, xy, xz], axis=-1)[..., None, :],
            jnp.concatenate([yx, yy, yz], axis=-1)[..., None, :],
            jnp.concatenate([zx, zy, zz], axis=-1)[..., None, :],
        ],
        axis=-2,
    )


def normalize(v: jnp.ndarray, axis: int = -1, eps: float = 1e-8) -> jnp.ndarray:
    norm = jnp.linalg.norm(v, axis=axis, keepdims=True)
    return v / jnp.clip(norm, a_min=eps)


def rotation6d_to_rotation_matrix(r6d: jnp.ndarray) -> jnp.ndarray:
    """Convert 6D rotation representation to rotation matrix.

    From https://arxiv.org/pdf/1812.07035, Appendix B

    Args:
        r6d: The 6D rotation representation, shape (*, 6).

    Returns:
        The rotation matrix, shape (*, 3, 3).
    """
    chex.assert_shape(r6d, (..., 6))
    shape = r6d.shape
    flat = r6d.reshape(-1, 6)
    a_1 = flat[:, 0:3]
    a_2 = flat[:, 3:6]

    b_1 = normalize(a_1, axis=-1)

    # Reordered Gram-Schmidt orthonormalization.
    b_3 = normalize(jnp.cross(b_1, a_2), axis=-1)
    b_2 = jnp.cross(b_3, b_1)

    rotation_matrix = jnp.stack([b_1, b_2, b_3], axis=-1)
    return rotation_matrix.reshape(shape[:-1] + (3, 3))


def rotation_matrix_to_rotation6d(rotation_matrix: jnp.ndarray) -> jnp.ndarray:
    """Convert rotation matrix to 6D rotation representation.

    Args:
        rotation_matrix: The rotation matrix, shape (*, 3, 3).

    Returns:
        The 6D rotation representation, shape (*, 6).
    """
    chex.assert_shape(rotation_matrix, (..., 3, 3))
    shape = rotation_matrix.shape
    # Simply concatenate a1 and a2 from SO(3)
    r6d = jnp.concatenate([rotation_matrix[..., 0], rotation_matrix[..., 1]], axis=-1)
    return r6d.reshape(shape[:-2] + (6,))
