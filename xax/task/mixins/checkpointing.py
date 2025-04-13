"""Defines a mixin for handling model checkpointing."""

import io
import json
import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar, cast, overload

import equinox as eqx
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


Config = TypeVar("Config", bound=CheckpointingConfig)


class CheckpointingMixin(ArtifactsMixin[Config], Generic[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.__last_ckpt_time = 0.0

    def get_ckpt_path(self, state: State | None = None) -> Path:
        return get_ckpt_path(self.exp_dir, state)

    def get_init_ckpt_path(self) -> Path | None:
        if self._exp_dir is not None:
            if (ckpt_path := self.get_ckpt_path()).exists():
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
            last_time, cur_time = self.__last_ckpt_time, state.elapsed_time_s.item()
            if cur_time - last_time >= self.config.save_every_n_seconds:
                self.__last_ckpt_time = cur_time
                return True
        return False

    @overload
    def load_ckpt_with_template(
        self,
        path: Path,
        *,
        part: Literal["all"],
        model_template: PyTree,
        optimizer_template: PyTree,
        opt_state_template: PyTree,
    ) -> tuple[PyTree, optax.GradientTransformation, optax.OptState, State, Config]: ...

    @overload
    def load_ckpt_with_template(
        self,
        path: Path,
        *,
        part: Literal["model_state_config"],
        model_template: PyTree,
    ) -> tuple[PyTree, State, Config]: ...

    @overload
    def load_ckpt_with_template(
        self,
        path: Path,
        *,
        part: Literal["model"],
        model_template: PyTree,
    ) -> PyTree: ...

    @overload
    def load_ckpt_with_template(
        self,
        path: Path,
        *,
        part: Literal["opt"],
        optimizer_template: PyTree,
    ) -> optax.GradientTransformation: ...

    @overload
    def load_ckpt_with_template(
        self,
        path: Path,
        *,
        part: Literal["opt_state"],
        opt_state_template: PyTree,
    ) -> optax.OptState: ...

    @overload
    def load_ckpt_with_template(
        self,
        path: Path,
        *,
        part: Literal["state"],
    ) -> State: ...

    @overload
    def load_ckpt_with_template(
        self,
        path: Path,
        *,
        part: Literal["config"],
    ) -> Config: ...

    def load_ckpt_with_template(
        self,
        path: Path,
        *,
        part: CheckpointPart = "all",
        model_template: PyTree | None = None,
        optimizer_template: PyTree | None = None,
        opt_state_template: PyTree | None = None,
    ) -> (
        tuple[PyTree, optax.GradientTransformation, optax.OptState, State, Config]
        | tuple[PyTree, State, Config]
        | PyTree
        | optax.GradientTransformation
        | optax.OptState
        | State
        | Config
    ):
        """Load a checkpoint.

        Args:
            path: Path to the checkpoint directory
            part: Which part of the checkpoint to load
            model_template: Template model with correct structure but uninitialized weights
            optimizer_template: Template optimizer with correct structure but uninitialized weights
            opt_state_template: Template optimizer state with correct structure but uninitialized weights

        Returns:
            The requested checkpoint components
        """
        with tarfile.open(path, "r:gz") as tar:

            def get_model() -> PyTree:
                if model_template is None:
                    raise ValueError("model_template must be provided to load model weights")
                if (model := tar.extractfile("model")) is None:
                    raise ValueError(f"Checkpoint does not contain a model file: {path}")
                return eqx.tree_deserialise_leaves(io.BytesIO(model.read()), model_template)

            def get_opt() -> optax.GradientTransformation:
                if optimizer_template is None:
                    raise ValueError("optimizer_template must be provided to load optimizer")
                if (opt := tar.extractfile("optimizer")) is None:
                    raise ValueError(f"Checkpoint does not contain an optimizer file: {path}")
                return eqx.tree_deserialise_leaves(io.BytesIO(opt.read()), optimizer_template)

            def get_opt_state() -> optax.OptState:
                if opt_state_template is None:
                    raise ValueError("opt_state_template must be provided to load optimizer state")
                if (opt_state := tar.extractfile("opt_state")) is None:
                    raise ValueError(f"Checkpoint does not contain an optimizer state file: {path}")
                return eqx.tree_deserialise_leaves(io.BytesIO(opt_state.read()), opt_state_template)

            def get_state() -> State:
                if (state := tar.extractfile("state")) is None:
                    raise ValueError(f"Checkpoint does not contain a state file: {path}")
                return State.from_dict(**json.loads(state.read().decode()))

            def get_config() -> Config:
                if (config := tar.extractfile("config")) is None:
                    raise ValueError(f"Checkpoint does not contain a config file: {path}")
                return self.get_config(cast(DictConfig, OmegaConf.load(config)), use_cli=False)

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
        model: PyTree | None = None,
        optimizer: optax.GradientTransformation | None = None,
        opt_state: optax.OptState | None = None,
        aux_data: PyTree | None = None,
        state: State | None = None,
    ) -> Path:
        """Save a checkpoint.

        Args:
            model: The model to save
            state: The current training state
            optimizer: The optimizer to save
            aux_data: Additional data to save
            opt_state: The optimizer state to save

        Returns:
            Path to the saved checkpoint
        """
        ckpt_path = self.get_ckpt_path(state)

        if not is_master():
            return ckpt_path

        # Gets the path to the last checkpoint
        logger.info("Saving checkpoint to %s", ckpt_path)
        last_ckpt_path = self.get_ckpt_path()
        ckpt_path.parent.mkdir(exist_ok=True, parents=True)

        # Potentially removes the last checkpoint
        if last_ckpt_path.exists() and self.config.only_save_most_recent:
            if (base_ckpt := last_ckpt_path.resolve()).is_file():
                base_ckpt.unlink()

        # Save the checkpoint components
        with tarfile.open(ckpt_path, "w:gz") as tar:

            def add_file(name: str, buf: io.BytesIO) -> None:
                tarinfo = tarfile.TarInfo(name)
                tarinfo.size = buf.tell()
                buf.seek(0)
                tar.addfile(tarinfo, buf)

            # Save model using Equinox
            if model is not None:
                with io.BytesIO() as buf:
                    eqx.tree_serialise_leaves(buf, model)
                    add_file("model", buf)

            # Save optimizer using Equinox
            if optimizer is not None:
                with io.BytesIO() as buf:
                    eqx.tree_serialise_leaves(buf, optimizer)
                    add_file("optimizer", buf)

            # Save optimizer state using Equinox
            if opt_state is not None:
                with io.BytesIO() as buf:
                    eqx.tree_serialise_leaves(buf, opt_state)
                    add_file("opt_state", buf)

            # Save aux data using Equinox.
            if aux_data is not None:
                with io.BytesIO() as buf:
                    eqx.tree_serialise_leaves(buf, aux_data)
                    add_file("aux_data", buf)

            # Save state and config as JSON
            def add_file_bytes(name: str, data: bytes) -> None:  # noqa: ANN401
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            if state is not None:
                add_file_bytes("state", json.dumps(state.to_dict(), indent=2).encode())
            add_file_bytes("config", OmegaConf.to_yaml(self.config).encode())

        # Updates the symlink to the new checkpoint
        last_ckpt_path.unlink(missing_ok=True)
        try:
            last_ckpt_path.symlink_to(ckpt_path.relative_to(last_ckpt_path.parent))
        except FileExistsError:
            logger.exception("Exception while trying to update %s", ckpt_path)

        # Calls the base callback
        self.on_after_checkpoint_save(ckpt_path, state)

        return ckpt_path
