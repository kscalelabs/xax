"""Defines a mixin for handling model checkpointing."""

import io
import json
import logging
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Literal, TypeVar, cast, overload

import cloudpickle
import jax
import optax
from jaxtyping import PyTree
from omegaconf import DictConfig, OmegaConf

from xax.core.conf import field
from xax.core.state import State
from xax.nn.parallel import is_master
from xax.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin

logger = logging.getLogger(__name__)

CheckpointPart = Literal["model", "opt", "opt_state", "state", "config", "model_state_config", "all"]


def get_ckpt_path(exp_dir: Path, state: State | None = None) -> Path:
    """Defines the path to the checkpoint for a given state.

    Args:
        exp_dir: The experiment directory
        state: The current trainer state

    Returns:
        The path to the checkpoint file.
    """
    if state is None:
        return exp_dir / "checkpoints" / "ckpt.bin"
    return exp_dir / "checkpoints" / f"ckpt.{state.num_steps}.bin"


@jax.tree_util.register_dataclass
@dataclass
class CheckpointingConfig(ArtifactsConfig):
    save_every_n_steps: int | None = field(None, help="Save a checkpoint every N steps")
    save_every_n_seconds: float | None = field(60.0 * 60.0, help="Save a checkpoint every N seconds")
    only_save_most_recent: bool = field(True, help="Only keep the most recent checkpoint")
    load_from_ckpt_path: str | None = field(None, help="If set, load initial model weights from this path")
    load_ckpt_strict: bool = field(True, help="If set, only load weights for which have a matching key in the model")
    save_tf_model: bool = field(False, help="If set, saves a Tensorflow version of the model")


Config = TypeVar("Config", bound=CheckpointingConfig)


class CheckpointingMixin(ArtifactsMixin[Config], Generic[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.__last_ckpt_time = 0.0

    def get_ckpt_path(self, state: State | None = None) -> Path:
        return get_ckpt_path(self.exp_dir, state)

    def get_init_ckpt_path(self) -> Path | None:
        if self._exp_dir is not None:
            ckpt_path = self.get_ckpt_path()
            if ckpt_path.exists():
                return ckpt_path
        if self.config.load_from_ckpt_path is not None:
            ckpt_path = Path(self.config.load_from_ckpt_path)
            assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist."
            return ckpt_path
        return None

    def should_checkpoint(self, state: State) -> bool:
        if self.config.save_every_n_steps is not None:
            if state.num_steps % self.config.save_every_n_steps == 0:
                return True
        if self.config.save_every_n_seconds is not None:
            last_time, cur_time = self.__last_ckpt_time, state.elapsed_time_s
            if cur_time - last_time >= self.config.save_every_n_seconds:
                self.__last_ckpt_time = cur_time
                return True
        return False

    @overload
    def load_checkpoint(
        self,
        path: Path,
        part: Literal["all"] = "all",
    ) -> tuple[PyTree, optax.GradientTransformation, optax.OptState, State, DictConfig]: ...

    @overload
    def load_checkpoint(
        self,
        path: Path,
        part: Literal["model_state_config"] = "model_state_config",
    ) -> tuple[PyTree, State, DictConfig]: ...

    @overload
    def load_checkpoint(self, path: Path, part: Literal["model"]) -> PyTree: ...

    @overload
    def load_checkpoint(self, path: Path, part: Literal["opt"]) -> optax.GradientTransformation: ...

    @overload
    def load_checkpoint(self, path: Path, part: Literal["opt_state"]) -> optax.OptState: ...

    @overload
    def load_checkpoint(self, path: Path, part: Literal["state"]) -> State: ...

    @overload
    def load_checkpoint(self, path: Path, part: Literal["config"]) -> DictConfig: ...

    def load_checkpoint(
        self,
        path: Path,
        part: CheckpointPart = "all",
    ) -> (
        tuple[PyTree, optax.GradientTransformation, optax.OptState, State, DictConfig]
        | tuple[PyTree, State, DictConfig]
        | PyTree
        | optax.GradientTransformation
        | optax.OptState
        | State
        | DictConfig
    ):
        # Calls the base callback.
        self.on_before_checkpoint_load(path)

        with tarfile.open(path, "r:gz") as tar:

            def get_model() -> PyTree:
                if (model := tar.extractfile("model")) is None:
                    raise ValueError(f"Checkpoint does not contain a model file: {path}")
                return cloudpickle.load(model)

            def get_opt() -> optax.GradientTransformation:
                if (opt := tar.extractfile("opt")) is None:
                    raise ValueError(f"Checkpoint does not contain an opt file: {path}")
                return cloudpickle.load(opt)

            def get_opt_state() -> optax.OptState:
                if (opt_state := tar.extractfile("opt_state")) is None:
                    raise ValueError(f"Checkpoint does not contain an opt_state file: {path}")
                return cloudpickle.load(opt_state)

            def get_state() -> State:
                if (state := tar.extractfile("state")) is None:
                    raise ValueError(f"Checkpoint does not contain a state file: {path}")
                return State(**json.loads(state.read().decode()))

            def get_config() -> DictConfig:
                if (config := tar.extractfile("config")) is None:
                    raise ValueError(f"Checkpoint does not contain a config file: {path}")
                return cast(DictConfig, OmegaConf.load(config))

            match part:
                case "model":
                    return get_model()
                case "opt":
                    return get_opt()
                case "opt_state":
                    return get_opt_state()
                case "state":
                    return get_state()
                case "config":
                    return get_config()
                case "model_state_config":
                    return get_model(), get_state(), get_config()
                case "all":
                    return get_model(), get_opt(), get_opt_state(), get_state(), get_config()
                case _:
                    raise ValueError(f"Invalid checkpoint part: {part}")

    def save_checkpoint(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        state: State,
    ) -> Path:
        ckpt_path = self.get_ckpt_path(state)

        if not is_master():
            return ckpt_path

        # Gets the path to the last checkpoint.
        logger.info("Saving checkpoint to %s", ckpt_path)
        last_ckpt_path = self.get_ckpt_path()
        ckpt_path.parent.mkdir(exist_ok=True, parents=True)

        # Potentially removes the last checkpoint.
        if last_ckpt_path.exists() and self.config.only_save_most_recent:
            if (base_ckpt := last_ckpt_path.resolve()).is_file():
                base_ckpt.unlink()

        # Combines all temporary files into a single checkpoint TAR file.
        with tarfile.open(ckpt_path, "w:gz") as tar:

            def add_file(name: str, write_fn: Callable[[io.BytesIO], Any]) -> None:
                with io.BytesIO() as buf:
                    write_fn(buf)
                    tarinfo = tarfile.TarInfo(name)
                    tarinfo.size = buf.tell()
                    buf.seek(0)
                    tar.addfile(tarinfo, buf)

            add_file("model", lambda buf: cloudpickle.dump(model, buf))
            add_file("opt", lambda buf: cloudpickle.dump(optimizer, buf))
            add_file("opt_state", lambda buf: cloudpickle.dump(opt_state, buf))
            add_file("state", lambda buf: buf.write(json.dumps(asdict(state), indent=2).encode()))
            add_file("config", lambda buf: buf.write(OmegaConf.to_yaml(self.config).encode()))

        if self.config.save_tf_model:
            try:
                from jax.experimental import jax2tf
            except ModuleNotFoundError:
                raise ImportError("Tensorflow is not installed. Install it with `pip install tensorflow`")

            tf_model = jax2tf.convert(model)
            add_file("model.tf", lambda buf: cloudpickle.dump(tf_model, buf))

        # Updates the symlink to the new checkpoint.
        last_ckpt_path.unlink(missing_ok=True)
        try:
            last_ckpt_path.symlink_to(ckpt_path.relative_to(last_ckpt_path.parent))
        except FileExistsError:
            logger.exception("Exception while trying to update %s", ckpt_path)

        # Calls the base callback.
        self.on_after_checkpoint_save(ckpt_path, state)

        return ckpt_path
