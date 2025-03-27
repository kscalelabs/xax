"""Defines some utility functions for interfacing with Jax."""

import collections
import inspect
import logging
import os
import time
from functools import wraps
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, ParamSpec, Self, Sequence, TypeVar, cast

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

K = TypeVar("K")
V = TypeVar("V")


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


class HashableArray:
    def __init__(self, array: np.ndarray | jnp.ndarray) -> None:
        if not isinstance(array, (np.ndarray, jnp.ndarray)):
            raise ValueError(f"Expected np.ndarray or jnp.ndarray, got {type(array)}")
        self.array = array
        self._hash: int | None = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.array.tobytes())
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashableArray):
            return False
        return bool(jnp.array_equal(self.array, other.array))


def hashable_array(array: np.ndarray | jnp.ndarray) -> HashableArray:
    return HashableArray(array)


def _prepare_freeze(xs: Any) -> Any:  # noqa: ANN401
    """Deep copy unfrozen dicts to make the dictionary FrozenDict safe."""
    if isinstance(xs, FrozenDict):
        return xs._dict  # pylint: disable=protected-access
    if not isinstance(xs, dict):
        return xs
    return {key: _prepare_freeze(val) for key, val in xs.items()}


def _indent(x: str, num_spaces: int) -> str:
    indent_str = " " * num_spaces
    lines = x.split("\n")
    assert not lines[-1]
    return "\n".join(indent_str + line for line in lines[:-1]) + "\n"


class FrozenKeysView(collections.abc.KeysView[K]):
    def __repr__(self) -> str:
        return f"frozen_dict_keys({list(self)})"


class FrozenValuesView(collections.abc.ValuesView[V]):
    def __repr__(self) -> str:
        return f"frozen_dict_values({list(self)})"


@jax.tree_util.register_pytree_with_keys_class
class FrozenDict(Mapping[K, V]):
    """An immutable variant of the Python dict."""

    __slots__ = ("_dict", "_hash")

    def __init__(self, *args: Any, __unsafe_skip_copy__: bool = False, **kwargs: Any) -> None:  # noqa: ANN401
        # make sure the dict is as
        xs = dict(*args, **kwargs)
        if __unsafe_skip_copy__:
            self._dict = xs
        else:
            self._dict = _prepare_freeze(xs)

        self._hash: int | None = None

    def __getitem__(self, key: K) -> V:
        v = self._dict[key]
        if isinstance(v, dict):
            return FrozenDict(v)  # type: ignore[return-value]
        return v

    def __setitem__(self, key: K, value: V) -> None:
        raise ValueError("FrozenDict is immutable.")

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def __iter__(self) -> iter[K]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return self.pretty_repr()

    def __reduce__(self) -> tuple[type["FrozenDict[K, V]"], tuple[dict[K, V]]]:
        return FrozenDict, (self.unfreeze(),)

    def pretty_repr(self, num_spaces: int = 4) -> str:
        """Returns an indented representation of the nested dictionary."""

        def pretty_dict(x: Any) -> str:
            if not isinstance(x, dict):
                return repr(x)
            rep = ""
            for key, val in x.items():
                rep += f"{key}: {pretty_dict(val)},\n"
            if rep:
                return "{\n" + _indent(rep, num_spaces) + "}"
            else:
                return "{}"

        return f"FrozenDict({pretty_dict(self._dict)})"

    def __hash__(self) -> int:
        if self._hash is None:
            h = 0
            for key, value in self.items():
                h ^= hash((key, value))
            self._hash = h
        return self._hash

    def copy(self, add_or_replace: Mapping[K, V] = MappingProxyType({})) -> Self:
        return type(self)({**self, **unfreeze(add_or_replace)})  # type: ignore[arg-type]

    def keys(self) -> FrozenKeysView[K]:
        return FrozenKeysView(self)

    def values(self) -> FrozenValuesView[V]:
        return FrozenValuesView(self)

    def items(self) -> iter[tuple[K, V]]:
        for key in self._dict:
            yield (key, self[key])

    def pop(self, key: K) -> tuple["FrozenDict[K, V]", V]:
        value = self[key]
        new_dict = dict(self._dict)
        new_dict.pop(key)
        new_self = type(self)(new_dict)
        return new_self, value

    def unfreeze(self) -> dict[K, V]:
        return unfreeze(self)  # type: ignore[return-value]

    def tree_flatten_with_keys(self) -> tuple[tuple[tuple[jax.tree_util.DictKey, Any], ...], tuple[K, ...]]:
        sorted_keys = sorted(self._dict)
        return tuple([(jax.tree_util.DictKey(k), self._dict[k]) for k in sorted_keys]), tuple(sorted_keys)

    @classmethod
    def tree_unflatten(cls, keys: tuple[K, ...], values: tuple[Any, ...]) -> "FrozenDict[K, V]":
        return cls({k: v for k, v in zip(keys, values)}, __unsafe_skip_copy__=True)


def unfreeze(x: FrozenDict[K, V] | dict[str, Any]) -> dict[Any, Any]:  # noqa: ANN401
    if isinstance(x, FrozenDict):
        return jax.tree_util.tree_map(lambda y: y, x._dict)
    elif isinstance(x, dict):
        ys = {}
        for key, value in x.items():
            ys[key] = unfreeze(value)
        return ys
    else:
        return x
