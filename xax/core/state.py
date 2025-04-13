"""Defines a dataclass for keeping track of the current training state."""

import time
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
    return cast(Phase, ["train", "valid"][i])


class StateDict(TypedDict, total=False):
    num_steps: NotRequired[int]
    num_samples: NotRequired[int]
    num_valid_steps: NotRequired[int]
    num_valid_samples: NotRequired[int]
    start_time_s: NotRequired[float]
    elapsed_time_s: NotRequired[float]
    phase: NotRequired[Phase]
    _phase: NotRequired[int]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class State:
    _int64_arr: Array = field(MISSING, help="Internal array for storing int64 values")
    _float64_arr: Array = field(MISSING, help="Internal array for storing floating-point values")

    @property
    def num_steps(self) -> int:
        return self._int64_arr[0].item()

    @property
    def num_samples(self) -> int:
        return self._int64_arr[1].item()

    @property
    def num_valid_steps(self) -> int:
        return self._int64_arr[2].item()

    @property
    def num_valid_samples(self) -> int:
        return self._int64_arr[3].item()

    @property
    def start_time_s(self) -> float:
        return self._float64_arr[0].item()

    @property
    def elapsed_time_s(self) -> float:
        return self._float64_arr[1].item()

    @property
    def phase(self) -> Phase:
        return _int_to_phase(self._int64_arr[6])

    @classmethod
    def init_state(cls) -> "State":
        return cls(
            _int64_arr=jnp.array([0, 0, 0, 0, 0, 0, 0], dtype=jnp.int64),
            _float64_arr=jnp.array([time.time(), 0.0], dtype=jnp.float64),
        )

    @property
    def training(self) -> bool:
        return self.phase == "train"

    def num_phase_steps(self, phase: Phase) -> int:
        match phase:
            case "train":
                return self.num_steps
            case "valid":
                return self.num_valid_steps
            case _:
                raise ValueError(f"Invalid phase: {phase}")

    def replace(self, **kwargs: Unpack[StateDict]) -> "State":
        int64_arr = self._int64_arr
        float64_arr = self._float64_arr

        if "num_steps" in kwargs:
            int64_arr.at[0].set(kwargs["num_steps"])
        if "num_samples" in kwargs:
            int64_arr.at[1].set(kwargs["num_samples"])
        if "num_valid_steps" in kwargs:
            int64_arr.at[2].set(kwargs["num_valid_steps"])
        if "num_valid_samples" in kwargs:
            int64_arr.at[3].set(kwargs["num_valid_samples"])
        if "phase" in kwargs:
            int64_arr.at[6].set(_phase_to_int(kwargs["phase"]))
        if "_phase" in kwargs:
            int64_arr.at[6].set(kwargs["_phase"])

        if "start_time_s" in kwargs:
            float64_arr.at[0].set(kwargs["start_time_s"])
        if "elapsed_time_s" in kwargs:
            float64_arr.at[1].set(kwargs["elapsed_time_s"])

        return State(
            _int64_arr=int64_arr,
            _float64_arr=float64_arr,
        )

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "num_steps": int(self.num_steps),
            "num_samples": int(self.num_samples),
            "num_valid_steps": int(self.num_valid_steps),
            "num_valid_samples": int(self.num_valid_samples),
            "start_time_s": float(self.start_time_s),
            "elapsed_time_s": float(self.elapsed_time_s),
            "phase": str(self.phase),
        }

    @classmethod
    def from_dict(cls, **d: Unpack[StateDict]) -> "State":
        if "phase" in d:
            d["_phase"] = _phase_to_int(cast(Phase, d.pop("phase")))

        int64_arr = jnp.array(
            [
                d.get("num_steps", 0),
                d.get("num_samples", 0),
                d.get("num_valid_steps", 0),
                d.get("num_valid_samples", 0),
                d.get("_phase", 0),
            ],
            dtype=jnp.int64,
        )

        float64_arr = jnp.array(
            [
                d.get("start_time_s", time.time()),
                d.get("elapsed_time_s", 0.0),
            ],
            dtype=jnp.float64,
        )

        return cls(
            __int64_arr=int64_arr,
            __float64_arr=float64_arr,
        )
