"""Defines some useful Jax debugging utilities."""

from collections import deque
from collections.abc import Iterable, Mapping
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

        if hasattr(node, "__dict__") and isinstance(node.__dict__, Mapping):
            for cname, cnode in node.__dict__.items():
                gname = f"{name}.{cname}" if name else cname
                if is_leaf(cnode):
                    ret.append((gname, cnode))
                else:
                    q.append((gname, cnode))

        elif isinstance(node, Mapping):
            for cname, cnode in node.items():
                gname = f"{name}.{cname}" if name else cname
                if is_leaf(cnode):
                    ret.append((gname, cnode))
                else:
                    q.append((gname, cnode))

        elif isinstance(node, Iterable):
            for i, cnode in enumerate(node):
                gname = f"{name}.{i}" if name else str(i)
                if is_leaf(cnode):
                    ret.append((gname, cnode))
                else:
                    q.append((gname, cnode))

    return ret
