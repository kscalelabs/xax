"""Defines a mixin for incorporating some logging functionality."""

from enum import Enum
import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Generic, Self, TypeVar, Any

import jax

from xax.core.conf import field
from xax.core.state import State
from xax.task.base import BaseConfig, BaseTask
from xax.task.logger import Logger, LoggerImpl
from xax.task.loggers.json import JsonLogger
from xax.task.loggers.state import StateLogger
from xax.task.loggers.stdout import StdoutLogger
from xax.task.loggers.tensorboard import TensorboardLogger
from xax.task.loggers.wandb import WandbLogger, WandbConfigMode, WandbConfigResume
from xax.task.mixins.artifacts import ArtifactsMixin
from xax.utils.text import is_interactive_session


class LoggerBackend(str, Enum):
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"


@jax.tree_util.register_dataclass
@dataclass
class LoggerConfig(BaseConfig):
    log_interval_seconds: float = field(
        value=1.0,
        help="The interval between successive log lines.",
    )
    logger_backend: LoggerBackend = field(
        value=LoggerBackend.TENSORBOARD,
        help="The logger backend to use",
    )
    tensorboard_log_interval_seconds: float = field(
        value=10.0,
        help="The interval between successive Tensorboard log lines.",
    )
    wandb_project: str | None = field(
        value=None,
        help="The name of the W&B project to log to.",
    )
    wandb_entity: str | None = field(
        value=None,
        help="The W&B entity (team or user) to log to.",
    )
    wandb_name: str | None = field(
        value=None,
        help="The name of this run in W&B.",
    )
    wandb_tags: list[str] | None = field(
        value=None,
        help="List of tags for this W&B run.",
    )
    wandb_notes: str | None = field(
        value=None,
        help="Notes about this W&B run.",
    )
    wandb_log_interval_seconds: float = field(
        value=10.0,
        help="The interval between successive W&B log lines.",
    )
    wandb_mode: WandbConfigMode = field(
        value="online",
        help="Mode for wandb (online, offline, or disabled).",
    )
    wandb_resume: WandbConfigResume = field(
        value=False,
        help="Whether to resume a previous run. Can be a run ID string.",
    )
    wandb_reinit: bool = field(
        value=True,
        help="Whether to allow multiple wandb.init() calls in the same process.",
    )


Config = TypeVar("Config", bound=LoggerConfig)


def get_env_var(name: str, default: bool) -> bool:
    if name not in os.environ:
        return default
    return os.environ[name].strip() == "1"


class LoggerMixin(BaseTask[Config], Generic[Config]):
    logger: Logger

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.logger = Logger()

    def log_directory(self) -> Path | None:
        return None

    def add_logger(self, *logger: LoggerImpl) -> None:
        self.logger.add_logger(*logger)

    def set_loggers(self) -> None:
        self.add_logger(
            StdoutLogger(
                log_interval_seconds=self.config.log_interval_seconds,
            )
            if is_interactive_session()
            else JsonLogger(
                log_interval_seconds=self.config.log_interval_seconds,
            )
        )

        # If this is also an ArtifactsMixin, we should default add some
        # additional loggers which log data to the artifacts directory.
        if isinstance(self, ArtifactsMixin):
            self.add_logger(
                StateLogger(
                    run_directory=self.exp_dir,
                ),
                self._create_logger_backend(),
            )

    def _create_logger_backend(self):
        match self.config.logger_backend:
            case LoggerBackend.TENSORBOARD:
                return TensorboardLogger(
                    run_directory=self.exp_dir if isinstance(self, ArtifactsMixin) else "./",
                    log_interval_seconds=self.config.tensorboard_log_interval_seconds,
                )
            case LoggerBackend.WANDB:
                run_config = {}
                if hasattr(self.config, '__dict__'):
                    # Convert config to a serializable dictionary
                    wandb_config = self._config_to_dict(self.config)

                return WandbLogger(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    name=self.config.wandb_name,
                    run_directory=self.exp_dir if isinstance(self, ArtifactsMixin) else None,
                    config=run_config,
                    tags=self.config.wandb_tags,
                    notes=self.config.wandb_notes,
                    log_interval_seconds=self.config.wandb_log_interval_seconds,
                    reinit=self.config.wandb_reinit,
                    resume=self.config.wandb_resume,
                    mode=self.config.wandb_mode,
                )
            case _:
                # This shouldn't happen, as validation should take care of this
                raise Exception(f"Invalid logger_backend '{self.config.logger_backend}'")

    def _config_to_dict(self, config: Any) -> dict[str, Any]:
        """Convert a config object to a dictionary for W&B logging.
        
        Args:
            config: The configuration object to convert.
            
        Returns:
            A dictionary representation of the config.
        """
        if hasattr(config, '__dict__'):
            result = {}
            for key, value in config.__dict__.items():
                if not key.startswith('_'):
                    # Recursively convert nested configs
                    if hasattr(value, '__dict__'):
                        result[key] = self._config_to_dict(value)
                    elif isinstance(value, (list, tuple)):
                        # Handle lists/tuples that might contain configs
                        result[key] = [
                            self._config_to_dict(item) if hasattr(item, '__dict__') else item
                            for item in value
                        ]
                    elif isinstance(value, dict):
                        # Handle dicts that might contain configs
                        result[key] = {
                            k: self._config_to_dict(v) if hasattr(v, '__dict__') else v
                            for k, v in value.items()
                        }
                    else:
                        result[key] = value
            return result
        return config

    def write_logs(self, state: State) -> None:
        self.logger.write(state)

    def __enter__(self) -> Self:
        self.logger.__enter__()
        return self

    def __exit__(self, t: type[BaseException] | None, e: BaseException | None, tr: TracebackType | None) -> None:
        self.logger.__exit__(t, e, tr)
        return super().__exit__(t, e, tr)
