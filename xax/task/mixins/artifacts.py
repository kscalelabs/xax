"""Defines a mixin for storing any task artifacts."""

import functools
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Self, TypeVar

import jax

from xax.core.conf import field, get_run_dir
from xax.core.state import State
from xax.nn.parallel import is_master
from xax.task.base import BaseConfig, BaseTask
from xax.utils.experiments import stage_environment
from xax.utils.logging import LOG_STATUS
from xax.utils.text import show_info

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class ArtifactsConfig(BaseConfig):
    exp_dir: str | None = field(None, help="The fixed experiment directory")


Config = TypeVar("Config", bound=ArtifactsConfig)


class ArtifactsMixin(BaseTask[Config]):
    _exp_dir: Path | None

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._exp_dir = None

    @functools.cached_property
    def run_dir(self) -> Path:
        run_dir = get_run_dir()
        if run_dir is None:
            task_file = inspect.getfile(self.__class__)
            run_dir = Path(task_file).resolve().parent
        return run_dir / self.task_name

    @property
    def exp_dir(self) -> Path:
        return self.get_exp_dir()

    def set_exp_dir(self, exp_dir: str | Path) -> Self:
        self._exp_dir = Path(exp_dir).expanduser().resolve()
        return self

    def get_exp_dir(self) -> Path:
        if self._exp_dir is not None:
            return self._exp_dir

        if self.config.exp_dir is not None:
            exp_dir = Path(self.config.exp_dir).expanduser().resolve()
            exp_dir.mkdir(parents=True, exist_ok=True)
            self._exp_dir = exp_dir
            logger.log(LOG_STATUS, self._exp_dir)
            return self._exp_dir

        def get_exp_dir(run_id: int) -> Path:
            return self.run_dir / f"run_{run_id}"

        run_id = 0
        while (exp_dir := get_exp_dir(run_id)).is_dir():
            run_id += 1
        exp_dir.mkdir(exist_ok=True, parents=True)
        self._exp_dir = exp_dir.expanduser().resolve()
        logger.log(LOG_STATUS, self._exp_dir)
        return self._exp_dir

    @functools.lru_cache(maxsize=None)
    def stage_environment(self) -> Path | None:
        stage_dir = (self.exp_dir / "code").resolve()
        try:
            stage_environment(self, stage_dir)
        except Exception:
            logger.exception("Failed to stage environment!")
            return None
        return stage_dir

    def on_training_end(self, state: State) -> State:
        state = super().on_training_end(state)

        if is_master():
            if self._exp_dir is None:
                show_info("Exiting training job", important=True)
            else:
                show_info(f"Exiting training job for {self.exp_dir}", important=True)

        return state
