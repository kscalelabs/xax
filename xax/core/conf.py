"""Defines base configuration functions and utilities."""

import functools
import os
from dataclasses import dataclass, field as field_base
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
from omegaconf import II, MISSING, Container as OmegaConfContainer, OmegaConf

from xax.utils.text import show_error

FieldType = Any


def field(value: FieldType, **kwargs: str) -> FieldType:
    """Short-hand function for getting a config field.

    Args:
        value: The current field's default value.
        kwargs: Additional metadata fields to supply.

    Returns:
        The dataclass field.
    """
    metadata: dict[str, Any] = {}
    metadata.update(kwargs)

    if hasattr(value, "__call__"):
        return field_base(default_factory=value, metadata=metadata)
    if value.__class__.__hash__ is None:
        return field_base(default_factory=lambda: value, metadata=metadata)
    return field_base(default=value, metadata=metadata)


def is_missing(cfg: Any, key: str) -> bool:  # noqa: ANN401
    """Utility function for checking if a config key is missing.

    This is for cases when you are using a raw dataclass rather than an
    OmegaConf container but want to treat them the same way.

    Args:
        cfg: The config to check
        key: The key to check

    Returns:
        Whether or not the key is missing a value in the config
    """
    if isinstance(cfg, OmegaConfContainer):
        if OmegaConf.is_missing(cfg, key):
            return True
        if OmegaConf.is_interpolation(cfg, key):
            try:
                getattr(cfg, key)
                return False
            except Exception:
                return True
    if getattr(cfg, key) is MISSING:
        return True
    return False


@dataclass
class ErrorHandling:
    enabled: bool = field(True, help="Is error handling enabled?")
    maximum_exceptions: int = field(10, help="Maximum number of errors to encounter")
    backoff_after: int = field(5, help="Start to do a sleeping backoff after this many exceptions")
    sleep_backoff: float = field(0.1, help="Sleep backoff amount")
    sleep_backoff_power: float = field(2.0, help="How much to multiply backoff for each successive exception")
    log_full_exception: bool = field(False, help="Log the full exception message for each exception")
    flush_exception_summary_every: int = field(500, help="How often to flush exception summary")
    report_top_n_exception_types: int = field(5, help="Number of exceptions to summarize")
    exception_location_traceback_depth: int = field(3, help="Traceback length for the exception location")


@dataclass
class Logging:
    hide_third_party_logs: bool = field(True, help="If set, hide third-party logs")
    log_level: str = field("INFO", help="The logging level to use")


@dataclass
class Device:
    cpu: bool = field(True, help="Whether to use the CPU")
    gpu: bool = field(II("oc.env:USE_GPU,1"), help="Whether to use the GPU")
    metal: bool = field(II("oc.env:USE_METAL,1"), help="Whether to use the Apple Silicon accelerator")
    use_fp64: bool = field(False, help="Always use the 64-bit floating point type")
    use_fp32: bool = field(False, help="Always use the 32-bit floating point type")
    use_bf16: bool = field(False, help="Always use the 16-bit bfloat type")
    use_fp16: bool = field(False, help="Always use the 16-bit floating point type")


def parse_dtype(cfg: Device) -> jnp.dtype | None:
    if cfg.use_fp64:
        return jnp.float64
    if cfg.use_fp32:
        return jnp.float32
    if cfg.use_bf16:
        return jnp.bfloat16
    if cfg.use_fp16:
        return jnp.float16
    return None


@dataclass
class Triton:
    use_triton_if_available: bool = field(True, help="Use Triton if available")


@dataclass
class Experiment:
    default_random_seed: int = field(1337, help="The default random seed to use")


@dataclass
class Directories:
    run: str = field(II("oc.env:RUN_DIR"), help="The run directory")
    data: str = field(II("oc.env:DATA_DIR"), help="The data directory")
    pretrained_models: str = field(II("oc.env:MODEL_DIR"), help="The models directory")


@dataclass
class SlurmPartition:
    partition: str = field(MISSING, help="The partition name")
    num_nodes: int = field(1, help="The number of nodes to use")


@dataclass
class Slurm:
    launch: dict[str, SlurmPartition] = field({}, help="The available launch configurations")


@dataclass
class UserConfig:
    error_handling: ErrorHandling = field(ErrorHandling)
    logging: Logging = field(Logging)
    device: Device = field(Device)
    triton: Triton = field(Triton)
    experiment: Experiment = field(Experiment)
    directories: Directories = field(Directories)
    slurm: Slurm = field(Slurm)


def user_config_path() -> Path:
    xaxrc_path_raw = os.environ.get("XAXRC_PATH", "~/.xax.yml")
    xaxrc_path = Path(xaxrc_path_raw).expanduser()
    return xaxrc_path


@functools.lru_cache(maxsize=None)
def _load_user_config_cached() -> UserConfig:
    xaxrc_path = user_config_path()
    base_cfg = OmegaConf.structured(UserConfig)

    # Writes the config file.
    if xaxrc_path.exists():
        cfg = OmegaConf.merge(base_cfg, OmegaConf.load(xaxrc_path))
    else:
        show_error(f"No config file was found in {xaxrc_path}; writing one...", important=True)
        OmegaConf.save(base_cfg, xaxrc_path)
        cfg = base_cfg

    # Looks in the current directory for a config file.
    local_cfg_path = Path("xax.yml")
    if local_cfg_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(local_cfg_path))

    return cast(UserConfig, cfg)


def load_user_config() -> UserConfig:
    """Loads the ``~/.xax.yml`` configuration file.

    Returns:
        The loaded configuration.
    """
    return _load_user_config_cached()


def get_run_dir() -> Path | None:
    config = load_user_config().directories
    if is_missing(config, "run"):
        return None
    (run_dir := Path(config.run)).mkdir(parents=True, exist_ok=True)
    return run_dir


def get_data_dir() -> Path:
    config = load_user_config().directories
    if is_missing(config, "data"):
        raise RuntimeError(
            "The data directory has not been set! You should set it in your config file "
            f"in {user_config_path()} or set the DATA_DIR environment variable."
        )
    return Path(config.data)


def get_pretrained_models_dir() -> Path:
    config = load_user_config().directories
    if is_missing(config, "pretrained_models"):
        raise RuntimeError(
            "The data directory has not been set! You should set it in your config file "
            f"in {user_config_path()} or set the MODEL_DIR environment variable."
        )
    return Path(config.pretrained_models)
