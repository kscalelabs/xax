"""Defines some useful Jax debugging utilities."""

from collections import deque
from typing import Any, Callable, Deque

from jaxtyping import Array


def get_named_leaves(
    obj: Any,  # noqa: ANN401
    is_leaf: Callable[[Any], bool] = lambda x: isinstance(x, Array),  # noqa: ANN401
) -> list[tuple[str, Any]]:  # noqa: ANN401
    ret: list[tuple[str, Any]] = []
    q: Deque[tuple[str, Any]] = deque()  # noqa: ANN401
    q.append(("", obj))
    while q:
        name, node = q.popleft()
        if not hasattr(node, "__dict__"):
            continue
        for cname, cnode in node.__dict__.items():
            gname = f"{name}.{cname}" if name else cname
            if is_leaf(cnode):
                ret.append((gname, cnode))
            else:
                q.append((gname, cnode))
    return ret
