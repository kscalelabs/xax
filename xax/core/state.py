"""Defines a dataclass for keeping track of the current training state."""

import time
from dataclasses import asdict, dataclass
from typing import Any, Literal, NotRequired, TypedDict, Unpack, cast

import jax
from omegaconf import MISSING

from xax.core.conf import field

Phase = Literal["train", "valid"]


class StateDict(TypedDict, total=False):
    num_steps: NotRequired[int]
    num_samples: NotRequired[int]
    num_valid_steps: NotRequired[int]
    num_valid_samples: NotRequired[int]
    start_time_s: NotRequired[float]
    elapsed_time_s: NotRequired[float]
    phase: NotRequired[Phase]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class State:
    num_steps: int = field(MISSING, help="Number of steps so far")
    num_samples: int = field(MISSING, help="Number of sample so far")
    num_valid_steps: int = field(MISSING, help="Number of validation steps so far")
    num_valid_samples: int = field(MISSING, help="Number of validation samples so far")
    start_time_s: float = field(MISSING, help="Start time of training")
    elapsed_time_s: float = field(MISSING, help="Total elapsed time so far")
    _phase: int = field(MISSING, help="Current training phase")

    @property
    def phase(self) -> Phase:
        return cast(Phase, ["train", "valid"][self._phase])

    @classmethod
    def init_state(cls) -> "State":
        return cls(
            num_steps=0,
            num_samples=0,
            num_valid_steps=0,
            num_valid_samples=0,
            start_time_s=time.time(),
            elapsed_time_s=0.0,
            _phase=0,
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
        extra_kwargs: dict[str, Any] = {}  # noqa: ANN401
        if "phase" in kwargs:
            phase = kwargs.pop("phase")
            match phase:
                case "train":
                    extra_kwargs["_phase"] = 0
                case "valid":
                    extra_kwargs["_phase"] = 1
                case _:
                    raise ValueError(f"Invalid phase: {phase}")
        return State(**{**asdict(self), **kwargs, **extra_kwargs})
