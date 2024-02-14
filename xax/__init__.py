"""Defines the top-level xax API.

This package is structured so that all the important stuff can be accessed
without having to dig around through the internals. This is done by lazily
importing the module by name.

This file can be maintained by running the update script:

.. code-block:: bash

    python -m scripts.update_api --inplace
"""

__version__ = "0.0.3"

# This list shouldn't be modified by hand; instead, run the update script.
__all__ = [
    "UserConfig",
    "field",
    "get_data_dir",
    "get_pretrained_models_dir",
    "get_run_dir",
    "load_user_config",
    "State",
    "cast_phase",
    "BaseLauncher",
    "CliLauncher",
    "SingleProcessLauncher",
    "LogAudio",
    "LogImage",
    "LogLine",
    "LogVideo",
    "Logger",
    "LoggerImpl",
    "JsonLogger",
    "StateLogger",
    "StdoutLogger",
    "TensorboardLogger",
    "CPUStatsOptions",
    "DataloaderConfig",
    "GPUStatsOptions",
    "Script",
    "ScriptConfig",
    "Config",
    "Task",
    "collate",
    "collate_non_null",
    "BaseFileDownloader",
    "DataDownloader",
    "ModelDownloader",
    "check_md5",
    "check_sha256",
    "get_git_state",
    "get_state_dict_prefix",
    "get_training_code",
    "save_config",
    "ColoredFormatter",
    "configure_logging",
    "one_hot",
    "partial_flatten",
    "worker_chunk",
    "TextBlock",
    "colored",
    "format_datetime",
    "format_timedelta",
    "outlined",
    "render_text_blocks",
    "show_error",
    "show_warning",
    "uncolored",
    "wrapped",
]

__all__ += [
    "CollateMode",
    "Phase",
]

import os
from typing import TYPE_CHECKING

# If this flag is set, eagerly imports the entire package (not recommended).
IMPORT_ALL = int(os.environ.get("XAX_IMPORT_ALL", "0")) != 0

del os

# This dictionary is auto-generated and shouldn't be modified by hand; instead,
# run the update script.
NAME_MAP: dict[str, str] = {
    "UserConfig": "core.conf",
    "field": "core.conf",
    "get_data_dir": "core.conf",
    "get_pretrained_models_dir": "core.conf",
    "get_run_dir": "core.conf",
    "load_user_config": "core.conf",
    "State": "core.state",
    "cast_phase": "core.state",
    "BaseLauncher": "task.launchers.base",
    "CliLauncher": "task.launchers.cli",
    "SingleProcessLauncher": "task.launchers.single_process",
    "LogAudio": "task.logger",
    "LogImage": "task.logger",
    "LogLine": "task.logger",
    "LogVideo": "task.logger",
    "Logger": "task.logger",
    "LoggerImpl": "task.logger",
    "JsonLogger": "task.loggers.json",
    "StateLogger": "task.loggers.state",
    "StdoutLogger": "task.loggers.stdout",
    "TensorboardLogger": "task.loggers.tensorboard",
    "CPUStatsOptions": "task.mixins.cpu_stats",
    "DataLoaderConfig": "task.mixins.data_loader",
    "GPUStatsOptions": "task.mixins.gpu_stats",
    "Script": "task.script",
    "ScriptConfig": "task.script",
    "Config": "task.task",
    "Task": "task.task",
    "collate": "utils.data.collate",
    "collate_non_null": "utils.data.collate",
    "BaseFileDownloader": "utils.experiments",
    "DataDownloader": "utils.experiments",
    "ModelDownloader": "utils.experiments",
    "check_md5": "utils.experiments",
    "check_sha256": "utils.experiments",
    "get_git_state": "utils.experiments",
    "get_state_dict_prefix": "utils.experiments",
    "get_training_code": "utils.experiments",
    "save_config": "utils.experiments",
    "ColoredFormatter": "utils.logging",
    "configure_logging": "utils.logging",
    "one_hot": "utils.numpy",
    "partial_flatten": "utils.numpy",
    "worker_chunk": "utils.numpy",
    "TextBlock": "utils.text",
    "colored": "utils.text",
    "format_datetime": "utils.text",
    "format_timedelta": "utils.text",
    "outlined": "utils.text",
    "render_text_blocks": "utils.text",
    "show_error": "utils.text",
    "show_warning": "utils.text",
    "uncolored": "utils.text",
    "wrapped": "utils.text",
}

# Need to manually set some values which can't be auto-generated.
NAME_MAP.update(
    {
        "CollateMode": "utils.data.collate",
        "Phase": "core.state",
    },
)


def __getattr__(name: str) -> object:
    if name not in NAME_MAP:
        raise AttributeError(f"{__name__} has no attribute {name}")

    module_name = f"xax.{NAME_MAP[name]}"
    module = __import__(module_name, fromlist=[name])
    return getattr(module, name)


if IMPORT_ALL or TYPE_CHECKING:
    from xax.core.conf import (
        UserConfig,
        field,
        get_data_dir,
        get_pretrained_models_dir,
        get_run_dir,
        load_user_config,
    )
    from xax.core.state import Phase, State, cast_phase
    from xax.task.launchers.base import BaseLauncher
    from xax.task.launchers.cli import CliLauncher
    from xax.task.launchers.single_process import SingleProcessLauncher
    from xax.task.logger import LogAudio, Logger, LoggerImpl, LogImage, LogLine, LogVideo
    from xax.task.loggers.json import JsonLogger
    from xax.task.loggers.state import StateLogger
    from xax.task.loggers.stdout import StdoutLogger
    from xax.task.loggers.tensorboard import TensorboardLogger
    from xax.task.mixins.cpu_stats import CPUStatsOptions
    from xax.task.mixins.data_loader import DataloaderConfig
    from xax.task.mixins.gpu_stats import GPUStatsOptions
    from xax.task.script import Script, ScriptConfig
    from xax.task.task import Config, Task
    from xax.utils.data.collate import CollateMode, collate, collate_non_null
    from xax.utils.experiments import (
        BaseFileDownloader,
        DataDownloader,
        ModelDownloader,
        check_md5,
        check_sha256,
        get_git_state,
        get_state_dict_prefix,
        get_training_code,
        save_config,
    )
    from xax.utils.logging import ColoredFormatter, configure_logging
    from xax.utils.numpy import one_hot, partial_flatten, worker_chunk
    from xax.utils.text import (
        TextBlock,
        colored,
        format_datetime,
        format_timedelta,
        outlined,
        render_text_blocks,
        show_error,
        show_warning,
        uncolored,
        wrapped,
    )

del TYPE_CHECKING, IMPORT_ALL
