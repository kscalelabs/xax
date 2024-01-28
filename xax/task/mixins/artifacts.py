"""Defines a mixin for storing any task artifacts."""

import functools
import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self, TypeVar

from mlfab.core.conf import field, get_run_dir
from mlfab.core.state import State
from mlfab.nn.parallel import is_master
from mlfab.task.base import BaseConfig, BaseTask
from mlfab.utils.experiments import add_toast, stage_environment
from mlfab.utils.text import show_info

logger = logging.getLogger(__name__)


@dataclass
class ArtifactsConfig(BaseConfig):
    exp_dir: str | None = field(None, help="The fixed experiment directory")


Config = TypeVar("Config", bound=ArtifactsConfig)


class ArtifactsMixin(BaseTask[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._exp_dir: Path | None = None

    @functools.cached_property
    def run_dir(self) -> Path:
        run_dir = get_run_dir()
        if run_dir is None:
            task_file = inspect.getfile(self.__class__)
            run_dir = Path(task_file).resolve().parent
        return run_dir / self.task_name

    def add_lock_file(self, lock_type: str, *, exists_ok: bool = False) -> None:
        if (lock_file := self.exp_dir / f".lock_{lock_type}").exists():
            if not exists_ok:
                raise RuntimeError(f"Lock file already exists at {lock_file}")
        else:
            with open(lock_file, "w", encoding="utf-8") as f:
                f.write(f"PID: {os.getpid()}")

    def remove_lock_file(self, lock_type: str, *, missing_ok: bool = False) -> None:
        if (lock_file := self.exp_dir / f".lock_{lock_type}").exists():
            lock_file.unlink()
        elif not missing_ok:
            raise RuntimeError(f"Lock file not found at {lock_file}")

    def get_exp_dir(self) -> Path:
        if self._exp_dir is not None:
            return self._exp_dir

        if self.config.exp_dir is not None:
            exp_dir = Path(self.config.exp_dir).expanduser().resolve()
            exp_dir.mkdir(parents=True, exist_ok=True)
            self._exp_dir = exp_dir
            add_toast("status", self._exp_dir)
            return self._exp_dir

        def get_exp_dir(run_id: int) -> Path:
            return self.run_dir / f"run_{run_id}"

        def has_lock_file(exp_dir: Path, lock_type: str | None = None) -> bool:
            if lock_type is not None:
                return (exp_dir / f".lock_{lock_type}").exists()
            return any(exp_dir.glob(".lock_*"))

        run_id = 0
        while (exp_dir := get_exp_dir(run_id)).is_dir() and has_lock_file(exp_dir):
            run_id += 1
        exp_dir.mkdir(exist_ok=True, parents=True)
        self._exp_dir = exp_dir.expanduser().resolve()
        add_toast("status", self._exp_dir)
        return self._exp_dir

    def set_exp_dir(self, exp_dir: Path) -> Self:
        self._exp_dir = exp_dir
        return self

    @property
    def exp_dir(self) -> Path:
        return self.get_exp_dir()

    @functools.lru_cache(maxsize=None)
    def stage_environment(self) -> Path | None:
        stage_dir = (self.exp_dir / "code").resolve()
        try:
            stage_environment(self, stage_dir)
        except Exception:
            logger.exception("Failed to stage environment!")
            return None
        return stage_dir

    def on_training_end(self, state: State) -> None:
        super().on_training_end(state)

        if is_master():
            if self._exp_dir is None:
                show_info("Exiting training job", important=True)
            else:
                show_info(f"Exiting training job for {self.exp_dir}", important=True)
