"""Defines a dataclass for keeping track of the current training state."""

import time
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict, cast, get_args

from omegaconf import MISSING

from xax.core.conf import field

Phase = Literal["train", "valid"]


def cast_phase(raw_phase: str) -> Phase:
    args = get_args(Phase)
    assert raw_phase in args, f"Invalid phase: '{raw_phase}' Valid options are {args}"
    return cast(Phase, raw_phase)


class StateDict(TypedDict, total=False):
    num_steps: NotRequired[int]
    num_samples: NotRequired[int]
    num_valid_steps: NotRequired[int]
    num_valid_samples: NotRequired[int]
    start_time_s: NotRequired[float]
    elapsed_time_s: NotRequired[float]
    raw_phase: NotRequired[str]


@dataclass
class State:
    num_steps: int = field(MISSING, help="Number of steps so far")
    num_samples: int = field(MISSING, help="Number of sample so far")
    num_valid_steps: int = field(MISSING, help="Number of validation steps so far")
    num_valid_samples: int = field(MISSING, help="Number of validation samples so far")
    start_time_s: float = field(MISSING, help="Start time of training")
    elapsed_time_s: float = field(MISSING, help="Total elapsed time so far")
    raw_phase: str = field(MISSING, help="Current training phase")

    @property
    def phase(self) -> Phase:
        return cast_phase(self.raw_phase)

    @phase.setter
    def phase(self, phase: Phase) -> None:
        self.raw_phase = phase

    @classmethod
    def init_state(cls) -> "State":
        return cls(
            num_steps=0,
            num_samples=0,
            num_valid_steps=0,
            num_valid_samples=0,
            start_time_s=time.time(),
            elapsed_time_s=0.0,
            raw_phase="train",
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
