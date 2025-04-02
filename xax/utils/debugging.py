"""Defines some useful Jax debugging utilities."""

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Deque

import jax
import jax.numpy as jnp
from jaxtyping import Array


def get_named_leaves(
    obj: Any,  # noqa: ANN401
    is_leaf: Callable[[Any], bool] = lambda x: isinstance(x, Array),  # noqa: ANN401
    max_depth: int = 100,
) -> list[tuple[str, Any]]:  # noqa: ANN401
    ret: list[tuple[str, Any]] = []
    q: Deque[tuple[int, str, Any]] = deque()  # noqa: ANN401
    q.append((0, "", obj))

    while q:
        depth, name, node = q.popleft()

        if depth > max_depth:
            continue

        if hasattr(node, "__dict__") and isinstance(node.__dict__, Mapping):
            for cname, cnode in node.__dict__.items():
                gname = f"{name}.{cname}" if name else cname
                if is_leaf(cnode):
                    ret.append((gname, cnode))
                else:
                    q.append((depth + 1, gname, cnode))

        elif isinstance(node, Mapping):
            for cname, cnode in node.items():
                gname = f"{name}.{cname}" if name else cname
                if is_leaf(cnode):
                    ret.append((gname, cnode))
                else:
                    q.append((depth + 1, gname, cnode))

        elif isinstance(node, Iterable):
            for i, cnode in enumerate(node):
                gname = f"{name}.{i}" if name else str(i)
                if is_leaf(cnode):
                    ret.append((gname, cnode))
                else:
                    q.append((depth + 1, gname, cnode))

    return ret


def breakpoint_if_nan(x: Array) -> None:
    jax.lax.cond(jnp.any(jnp.isnan(x)), lambda: jax.debug.breakpoint(), lambda: None)


def log_if_nan(x: Array, loc: str) -> None:
    jax.lax.cond(jnp.any(jnp.isnan(x)), lambda: jax.debug.print("=== NaNs: {loc} ===", loc=loc), lambda: None)
