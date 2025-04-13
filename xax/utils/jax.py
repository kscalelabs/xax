"""Defines some utility functions for interfacing with Jax."""

import functools
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

# For control flow functions.
Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


@functools.lru_cache(maxsize=None)
def disable_jit_level() -> int:
    """Gets a debugging flag for disabling jitting.

    For Xax's JIT'ed functions, we can set a JIT level which can be used to
    disable jitting when we want to debug some NaN issues.

    Returns:
        The JIT level to disable.
    """
    return int(os.environ.get("DISABLE_JIT_LEVEL", "0"))


def should_disable_jit(jit_level: int | None) -> bool:
    return jit_level is not None and jit_level < disable_jit_level()


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
    jit_level: int | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrapper function that provides utility improvements over Jax's JIT.

    Specifically, this function works on class methods, is toggleable, and
    detects recompilations by matching hash values.

    This is meant to be used as a decorator factory, and the decorated function
    calls `wrapped`.
    """
    if should_disable_jit(jit_level):
        return lambda fn: fn  # Identity function.

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


def scan(
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X | None = None,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    jit_level: int | None = None,
) -> tuple[Carry, Y]:
    """A wrapper around jax.lax.scan that allows for more flexible tracing.

    If the provided JIT level is below the environment JIT level, we manually
    unroll the scan function as a for loop.

    Args:
        f: The function to scan.
        init: The initial value for the scan.
        xs: The input to the scan.
        length: The length of the scan.
        reverse: Whether to reverse the scan.
        unroll: The unroll factor for the scan.
        jit_level: The JIT level to use for the scan.

    Returns:
        A tuple containing the final carry and the output of the scan.
    """
    if not should_disable_jit(jit_level):
        return jax.lax.scan(f, init, xs, length, reverse, unroll)

    if xs is None:
        if length is None:
            raise ValueError("length must be provided if xs is None")
        xs = cast(X, [None] * length)

    carry = init
    ys = []
    for x in cast(Iterable, xs):
        carry, y = f(carry, x)
        ys.append(y)

    return carry, jax.tree.map(lambda *ys: jnp.stack(ys), *ys)
