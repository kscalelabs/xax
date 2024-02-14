"""Defines a base class with utility functions for staged training runs."""

from abc import ABC
from pathlib import Path

from xax.task.launchers.base import BaseLauncher
from xax.task.mixins.artifacts import ArtifactsMixin, Config


class StagedLauncher(BaseLauncher, ABC):
    def __init__(self, config_file_name: str = "config.yaml") -> None:
        super().__init__()

        self.config_file_name = config_file_name

    def get_config_path(self, task: "ArtifactsMixin[Config]", use_cli: bool | list[str] = True) -> Path:
        config_path = task.exp_dir / self.config_file_name
        task.config.exp_dir = str(task.exp_dir)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(task.config_str(task.config, use_cli=use_cli))
        return config_path

    @classmethod
    def from_components(cls, task_key: str, config_path: Path, use_cli: bool | list[str] = True) -> "ArtifactsMixin":
        return (
            ArtifactsMixin.from_task_key(task_key)
            .get_task(config_path, use_cli=use_cli)
            .set_exp_dir(config_path.parent)
        )
