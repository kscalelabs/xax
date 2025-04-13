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


def _clip_int(i: int) -> int:
    return max(min(i, 2**31 - 1), -(2**31))


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
    _int32_arr: Array = field(MISSING, help="Internal array for storing int64 values")
    _float32_arr: Array = field(MISSING, help="Internal array for storing floating-point values")

    @property
    def num_steps(self) -> int:
        return self._int32_arr[0].item()

    @property
    def num_samples(self) -> int:
        return self._int32_arr[1].item()

    @property
    def num_valid_steps(self) -> int:
        return self._int32_arr[2].item()

    @property
    def num_valid_samples(self) -> int:
        return self._int32_arr[3].item()

    @property
    def start_time_s(self) -> float:
        return self._float32_arr[0].item()

    @property
    def elapsed_time_s(self) -> float:
        return self._float32_arr[1].item()

    @property
    def phase(self) -> Phase:
        return _int_to_phase(self._int32_arr[6].item())

    @classmethod
    def init_state(cls) -> "State":
        return cls(
            _int32_arr=jnp.array([0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32),
            _float32_arr=jnp.array([time.time(), 0.0], dtype=jnp.float32),
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
        int32_arr = self._int32_arr
        float32_arr = self._float32_arr

        if "num_steps" in kwargs:
            int32_arr.at[0].set(_clip_int(kwargs["num_steps"]))
        if "num_samples" in kwargs:
            int32_arr.at[1].set(_clip_int(kwargs["num_samples"]))
        if "num_valid_steps" in kwargs:
            int32_arr.at[2].set(_clip_int(kwargs["num_valid_steps"]))
        if "num_valid_samples" in kwargs:
            int32_arr.at[3].set(_clip_int(kwargs["num_valid_samples"]))

        if "phase" in kwargs:
            int32_arr.at[6].set(_phase_to_int(kwargs["phase"]))
        if "_phase" in kwargs:
            int32_arr.at[6].set(kwargs["_phase"])

        if "start_time_s" in kwargs:
            float32_arr.at[0].set(kwargs["start_time_s"])
        if "elapsed_time_s" in kwargs:
            float32_arr.at[1].set(kwargs["elapsed_time_s"])

        return State(
            _int32_arr=int32_arr,
            _float32_arr=float32_arr,
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

        int32_arr = jnp.array(
            [
                _clip_int(d.get("num_steps", 0)),
                _clip_int(d.get("num_samples", 0)),
                _clip_int(d.get("num_valid_steps", 0)),
                _clip_int(d.get("num_valid_samples", 0)),
                _clip_int(d.get("_phase", 0)),
            ],
            dtype=jnp.int32,
        )

        float32_arr = jnp.array(
            [
                d.get("start_time_s", time.time()),
                d.get("elapsed_time_s", 0.0),
            ],
            dtype=jnp.float32,
        )

        return cls(
            _int32_arr=int32_arr,
            _float32_arr=float32_arr,
        )
