"""Defines a dataclass for keeping track of the current training state."""

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict, Unpack, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array
from omegaconf import MISSING

from xax.core.conf import field

Phase = Literal["train", "valid"]


def _phase_to_int(phase: Phase) -> int:
    return {"train": 0, "valid": 1}[phase]


def _int_to_phase(i: int) -> Phase:
    if i < 0 or i > 1:
        raise ValueError(f"Invalid phase: {i}")
    return cast(Phase, ["train", "valid"][i])


class StateDict(TypedDict, total=False):
    num_steps: NotRequired[int | Array]
    num_samples: NotRequired[int | Array]
    num_valid_steps: NotRequired[int | Array]
    num_valid_samples: NotRequired[int | Array]
    elapsed_time_s: NotRequired[float | Array]
    valid_elapsed_time_s: NotRequired[float | Array]
    phase: NotRequired[Phase]
    _phase: NotRequired[int | Array]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class State:
    _int32_arr: Array = field(MISSING, help="Internal array for storing int64 values")
    _float32_arr: Array = field(MISSING, help="Internal array for storing floating-point values")

    @property
    def num_steps(self) -> Array:
        return self._int32_arr[0]

    @property
    def num_valid_steps(self) -> Array:
        return self._int32_arr[1]

    @property
    def num_samples(self) -> Array:
        return self._float32_arr[0]

    @property
    def num_valid_samples(self) -> Array:
        return self._float32_arr[1]

    @property
    def elapsed_time_s(self) -> Array:
        return self._float32_arr[2]

    @property
    def valid_elapsed_time_s(self) -> Array:
        return self._float32_arr[3]

    @property
    def phase(self) -> Phase:
        return _int_to_phase(self._int32_arr[2].item())

    @classmethod
    def init_state(cls) -> "State":
        return cls(
            _int32_arr=jnp.array([0, 0, 0], dtype=jnp.int32),
            _float32_arr=jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        )

    @property
    def training(self) -> bool:
        return self.phase == "train"

    def replace(self, **kwargs: Unpack[StateDict]) -> "State":
        int32_arr = self._int32_arr
        float32_arr = self._float32_arr

        if "num_steps" in kwargs:
            int32_arr = int32_arr.at[0].set(kwargs["num_steps"])
        if "num_valid_steps" in kwargs:
            int32_arr = int32_arr.at[1].set(kwargs["num_valid_steps"])

        if "phase" in kwargs:
            int32_arr = int32_arr.at[2].set(_phase_to_int(kwargs["phase"]))
        if "_phase" in kwargs:
            int32_arr = int32_arr.at[2].set(kwargs["_phase"])

        if "num_samples" in kwargs:
            float32_arr = float32_arr.at[0].set(kwargs["num_samples"])
        if "num_valid_samples" in kwargs:
            float32_arr = float32_arr.at[1].set(kwargs["num_valid_samples"])

        if "elapsed_time_s" in kwargs:
            float32_arr = float32_arr.at[2].set(kwargs["elapsed_time_s"])
        if "valid_elapsed_time_s" in kwargs:
            float32_arr = float32_arr.at[3].set(kwargs["valid_elapsed_time_s"])

        return State(
            _int32_arr=int32_arr,
            _float32_arr=float32_arr,
        )

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "num_steps": int(self.num_steps),
            "num_valid_steps": int(self.num_valid_steps),
            "num_samples": int(self.num_samples),
            "num_valid_samples": int(self.num_valid_samples),
            "elapsed_time_s": float(self.elapsed_time_s),
            "valid_elapsed_time_s": float(self.valid_elapsed_time_s),
            "phase": str(self.phase),
        }

    @classmethod
    def from_dict(cls, **d: Unpack[StateDict]) -> "State":
        if "phase" in d:
            d["_phase"] = _phase_to_int(cast(Phase, d.pop("phase")))

        int32_arr = jnp.array(
            [
                d.get("num_steps", 0),
                d.get("num_valid_steps", 0),
                d.get("_phase", 0),
            ],
            dtype=jnp.int32,
        )

        float32_arr = jnp.array(
            [
                d.get("num_samples", 0),
                d.get("num_valid_samples", 0),
                d.get("elapsed_time_s", 0.0),
                d.get("valid_elapsed_time_s", 0.0),
            ],
            dtype=jnp.float32,
        )

        return cls(
            _int32_arr=int32_arr,
            _float32_arr=float32_arr,
        )
