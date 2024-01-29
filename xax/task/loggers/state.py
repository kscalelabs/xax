"""Defines a logger which logs the current training state."""

from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf

from xax.task.logger import LoggerImpl, LogLine


class StateLogger(LoggerImpl):
    def __init__(
        self,
        run_directory: str | Path,
        git_state_name: str = "git_state.txt",
        train_code_name: str = "train_code.py",
        config_name: str = "config.yaml",
        flush_immediately: bool = False,
        open_mode: Literal["w", "a"] = "w",
        line_sep: str = "\n",
        remove_unicode_from_namespaces: bool = True,
    ) -> None:
        super().__init__(float("inf"))

        self.git_state_file = Path(run_directory).expanduser().resolve() / git_state_name
        self.train_code_file = Path(run_directory).expanduser().resolve() / train_code_name
        self.config_file = Path(run_directory).expanduser().resolve() / config_name
        self.flush_immediately = flush_immediately
        self.open_mode = open_mode
        self.line_sep = line_sep
        self.remove_unicode_from_namespaces = remove_unicode_from_namespaces

    def log_git_state(self, git_state: str) -> None:
        with open(self.git_state_file, "w") as f:
            f.write(git_state)

    def log_training_code(self, training_code: str) -> None:
        with open(self.train_code_file, "w") as f:
            f.write(training_code)

    def log_config(self, config: DictConfig) -> None:
        OmegaConf.save(config, self.config_file)

    def write(self, line: LogLine) -> None:
        pass
