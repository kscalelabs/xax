"""Defines a mixin for handling model checkpointing."""

import io
import json
import logging
import tarfile
import tempfile
from dataclasses import asdict, dataclass
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
            ckpt_path = self.get_ckpt_path()
            if not ckpt_path.exists():
                logger.warning("No checkpoint found in experiment directory: %s", ckpt_path)
            else:
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
        *,
        part: Literal["all"],
        model_template: PyTree,
        optimizer_template: PyTree,
        opt_state_template: PyTree,
    ) -> tuple[PyTree, optax.GradientTransformation, optax.OptState, State, Config]: ...

    @overload
    def load_checkpoint(
        self,
        path: Path,
        *,
        part: Literal["model_state_config"],
        model_template: PyTree,
    ) -> tuple[PyTree, State, Config]: ...

    @overload
    def load_checkpoint(
        self,
        path: Path,
        *,
        part: Literal["model"],
        model_template: PyTree,
    ) -> PyTree: ...

    @overload
    def load_checkpoint(
        self,
        path: Path,
        *,
        part: Literal["opt"],
        optimizer_template: PyTree,
    ) -> optax.GradientTransformation: ...

    @overload
    def load_checkpoint(
        self,
        path: Path,
        *,
        part: Literal["opt_state"],
        opt_state_template: PyTree,
    ) -> optax.OptState: ...

    @overload
    def load_checkpoint(
        self,
        path: Path,
        *,
        part: Literal["state"],
    ) -> State: ...

    @overload
    def load_checkpoint(
        self,
        path: Path,
        *,
        part: Literal["config"],
    ) -> Config: ...

    def load_checkpoint(
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
        with tarfile.open(path, "r:gz") as tar, tempfile.TemporaryDirectory() as tempdir:

            def get_model() -> PyTree:
                if model_template is None:
                    raise ValueError("model_template must be provided to load model weights")
                if (model := tar.extractfile("model.eqx")) is None:
                    raise ValueError(f"Checkpoint does not contain a model file: {path}")
                # Create a temporary file to store the model data
                with (Path(tempdir) / "model.eqx").open("wb") as f:
                    f.write(model.read())
                # Use Equinox to deserialize the model
                return eqx.tree_deserialise_leaves(Path(tempdir) / "model.eqx", model_template)

            def get_opt() -> optax.GradientTransformation:
                if optimizer_template is None:
                    raise ValueError("optimizer_template must be provided to load optimizer")
                if (opt := tar.extractfile("optimizer.eqx")) is None:
                    raise ValueError(f"Checkpoint does not contain an optimizer file: {path}")
                # Create a temporary file to store the optimizer data
                with (Path(tempdir) / "optimizer.eqx").open("wb") as f:
                    f.write(opt.read())
                # Use Equinox to deserialize the optimizer
                return eqx.tree_deserialise_leaves(Path(tempdir) / "optimizer.eqx", optimizer_template)

            def get_opt_state() -> optax.OptState:
                if opt_state_template is None:
                    raise ValueError("opt_state_template must be provided to load optimizer state")
                if (opt_state := tar.extractfile("opt_state.eqx")) is None:
                    raise ValueError(f"Checkpoint does not contain an optimizer state file: {path}")
                # Create a temporary file to store the optimizer state data
                with (Path(tempdir) / "opt_state.eqx").open("wb") as f:
                    f.write(opt_state.read())
                return eqx.tree_deserialise_leaves(Path(tempdir) / "opt_state.eqx", opt_state_template)

            def get_state() -> State:
                if (state := tar.extractfile("state")) is None:
                    raise ValueError(f"Checkpoint does not contain a state file: {path}")
                return State(**json.loads(state.read().decode()))

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
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        state: State,
    ) -> Path:
        """Save a checkpoint.

        Args:
            model: The model to save
            optimizer: The optimizer to save
            opt_state: The optimizer state to save
            state: The current training state

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
        with tarfile.open(ckpt_path, "w:gz") as tar, tempfile.TemporaryDirectory() as tempdir:
            # Save model using Equinox
            eqx.tree_serialise_leaves(Path(tempdir) / "model.eqx", model)
            tar.add(Path(tempdir) / "model.eqx", "model.eqx")
            (Path(tempdir) / "model.eqx").unlink()

            # Save optimizer using cloudpickle
            eqx.tree_serialise_leaves(Path(tempdir) / "optimizer.eqx", optimizer)
            tar.add(Path(tempdir) / "optimizer.eqx", "optimizer.eqx")
            (Path(tempdir) / "optimizer.eqx").unlink()

            # Save optimizer state using Equinox
            eqx.tree_serialise_leaves(Path(tempdir) / "opt_state.eqx", opt_state)
            tar.add(Path(tempdir) / "opt_state.eqx", "opt_state.eqx")
            (Path(tempdir) / "opt_state.eqx").unlink()

            # Save state and config as JSON
            def add_file(name: str, data: bytes) -> None:  # noqa: ANN401
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            add_file("state", json.dumps(asdict(state), indent=2).encode())
            add_file("config", OmegaConf.to_yaml(self.config).encode())

        # Updates the symlink to the new checkpoint
        last_ckpt_path.unlink(missing_ok=True)
        try:
            last_ckpt_path.symlink_to(ckpt_path.relative_to(last_ckpt_path.parent))
        except FileExistsError:
            logger.exception("Exception while trying to update %s", ckpt_path)

        # Calls the base callback
        self.on_after_checkpoint_save(ckpt_path, state)

        return ckpt_path
