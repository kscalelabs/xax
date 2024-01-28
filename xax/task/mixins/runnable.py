"""Defines a mixin which provides a "run" method."""

import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import FrameType
from typing import Callable, TypeVar

from mlfab.task.base import BaseConfig, BaseTask, RawConfigType
from mlfab.task.launchers.base import BaseLauncher


@dataclass
class RunnableConfig(BaseConfig):
    pass


Config = TypeVar("Config", bound=RunnableConfig)


class RunnableMixin(BaseTask[Config], ABC):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.__signal_handlers: dict[signal.Signals, list[Callable[[], None]]] = {}
        self.__set_signal_handlers: set[signal.Signals] = set()

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
            from mlfab.task.launchers.cli import CliLauncher

            launcher = CliLauncher()
        launcher.launch(cls, *cfgs, use_cli=use_cli)

    def call_signal_handler(self, sig: int | signal.Signals, frame: FrameType | None = None) -> None:
        if isinstance(sig, int):
            sig = signal.Signals(sig)
        for signal_handler in self.__signal_handlers.get(sig, []):
            signal_handler()

    def add_signal_handler(self, handler: Callable[[], None], *sigs: signal.Signals) -> None:
        for sig in sigs:
            if sig not in self.__signal_handlers:
                self.__signal_handlers[sig] = []
            if sig not in self.__set_signal_handlers:
                self.__set_signal_handlers.add(sig)
                signal.signal(sig, self.call_signal_handler)
            self.__signal_handlers[sig].append(handler)
