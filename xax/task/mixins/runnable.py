"""Defines a mixin which provides a "run" method."""

import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import FrameType
from typing import Callable, TypeVar

import jax

from xax.task.base import BaseConfig, BaseTask, RawConfigType
from xax.task.launchers.base import BaseLauncher


@jax.tree_util.register_dataclass
@dataclass
class RunnableConfig(BaseConfig):
    pass


Config = TypeVar("Config", bound=RunnableConfig)


class RunnableMixin(BaseTask[Config], ABC):
    """Mixin which provides a "run" method."""

    _signal_handlers: dict[signal.Signals, list[Callable[[], None]]]
    _set_signal_handlers: set[signal.Signals]

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._signal_handlers = {}
        self._set_signal_handlers = set()

    @abstractmethod
    def run(self) -> None:
        """Runs the task."""

    @classmethod
    def launch(
        cls,
        *cfgs: RawConfigType,
        launcher: BaseLauncher | None = None,
        use_cli: bool | list[str] = True,
    ) -> None:
        if launcher is None:
            from xax.task.launchers.cli import CliLauncher

            launcher = CliLauncher()
        launcher.launch(cls, *cfgs, use_cli=use_cli)

    def call_signal_handler(self, sig: int | signal.Signals, frame: FrameType | None = None) -> None:
        if isinstance(sig, int):
            sig = signal.Signals(sig)
        for signal_handler in self._signal_handlers.get(sig, []):
            signal_handler()

    def add_signal_handler(self, handler: Callable[[], None], *sigs: signal.Signals) -> None:
        for sig in sigs:
            if sig not in self._signal_handlers:
                self._signal_handlers[sig] = []
            if sig not in self._set_signal_handlers:
                self._set_signal_handlers.add(sig)
                signal.signal(sig, self.call_signal_handler)
            self._signal_handlers[sig].append(handler)
