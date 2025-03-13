"""Defines some utility functions for interfacing with Jax."""

import inspect
import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Iterable, ParamSpec, Sequence, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import sharding_impls
from jax._src.lib import xla_client as xc

logger = logging.getLogger(__name__)

DEFAULT_COMPILE_TIMEOUT = 1.0

Number = int | float | np.ndarray | jnp.ndarray


P = ParamSpec("P")  # For function parameters
R = TypeVar("R")  # For function return type


def as_float(value: int | float | np.ndarray | jnp.ndarray) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return float(value.item())
    raise TypeError(f"Unexpected type: {type(value)}")


def get_hash(obj: object) -> int:
    """Get a hash of an object.

    If the object is hashable, use the hash. Otherwise, use the id.
    """
    if hasattr(obj, "__hash__"):
        return hash(obj)
    return id(obj)


def jit(
    in_shardings: Any = sharding_impls.UNSPECIFIED,  # noqa: ANN401
    out_shardings: Any = sharding_impls.UNSPECIFIED,  # noqa: ANN401
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: xc.Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    abstracted_axes: Any | None = None,  # noqa: ANN401
    compiler_options: dict[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrapper function that provides utility improvements over Jax's JIT.

    Specifically, this function works on class methods, is toggleable, and
    detects recompilations by matching hash values.

    This is meant to be used as a decorator factory, and the decorated function
    calls `wrapped`.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        class JitState:
            compilation_count = 0
            last_arg_dict: dict[str, int] | None = None

        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        jitted_fn = jax.jit(
            fn,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            donate_argnums=donate_argnums,
            donate_argnames=donate_argnames,
            keep_unused=keep_unused,
            device=device,
            backend=backend,
            inline=inline,
            abstracted_axes=abstracted_axes,
            compiler_options=compiler_options,
        )

        @wraps(fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            if os.environ.get("DEBUG", "0") == "1":  # skipping during debug
                return fn(*args, **kwargs)

            do_profile = os.environ.get("JIT_PROFILE", "0") == "1"

            if do_profile:
                class_name = (args[0].__class__.__name__) + "." if fn.__name__ == "__call__" else ""
                logger.info(
                    "Currently running %s (count: %s)",
                    f"{class_name}{fn.__name__}",
                    JitState.compilation_count,
                )

            start_time = time.time()
            res = jitted_fn(*args, **kwargs)
            end_time = time.time()
            runtime = end_time - start_time

            # if this is true, if runtime is higher than COMPILE_TIMEOUT, we recompile
            # TODO: we should probably reimplement the lower-level jitting logic to avoid this
            if do_profile:
                arg_dict = {}
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        arg_dict[param_names[i]] = get_hash(arg)
                for k, v in kwargs.items():
                    arg_dict[k] = get_hash(v)

                logger.info("Hashing took %s seconds", runtime)
                JitState.compilation_count += 1

                if JitState.last_arg_dict is not None:
                    all_keys = set(arg_dict.keys()) | set(JitState.last_arg_dict.keys())
                    for k in all_keys:
                        prev = JitState.last_arg_dict.get(k, "N/A")
                        curr = arg_dict.get(k, "N/A")

                        if prev != curr:
                            logger.info("- Arg '%s' hash changed: %s -> %s", k, prev, curr)

                JitState.last_arg_dict = arg_dict

            return cast(R, res)

        return wrapped

    return decorator


def quat_to_euler(quat_4: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Normalizes and converts a quaternion (w, x, y, z) to roll, pitch, yaw.

    Args:
        quat_4: The quaternion to convert, shape (*, 4).
        eps: A small epsilon value to avoid division by zero.

    Returns:
        The roll, pitch, yaw angles with shape (*, 3).
    """
    quat_4 = quat_4 / (jnp.linalg.norm(quat_4) + eps)
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

    return jnp.stack([roll, pitch, yaw], axis=-1)
