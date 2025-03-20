"""Defines some useful Jax debugging utilities."""

from collections import deque
from typing import Deque

import equinox as eqx
from jaxtyping import Array


def eqx_weight_names(model: eqx.Module) -> list[tuple[str, Array]]:
    ret: list[tuple[str, Array]] = []
    q: Deque[tuple[str, eqx.Module]] = deque()
    q.append(("", model))
    while q:
        name, node = q.popleft()
        for cname, cnode in node.__dict__.items():
            gname = f"{name}.{cname}" if name else cname
            if isinstance(cnode, eqx.Module):
                q.append((gname, cnode))
            elif isinstance(cnode, Array):
                ret.append((gname, cnode))
    return ret
