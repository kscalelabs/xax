"""Defines a mixin for handling model checkpointing."""

import io
import json
import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, Self, Sequence, TypeVar, cast, overload

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


Config = TypeVar("Config", bound=CheckpointingConfig)


@overload
def load_ckpt(
    path: Path,
    *,
    part: Literal["all"],
    model_templates: Sequence[PyTree],
    optimizer_templates: Sequence[optax.GradientTransformation],
    opt_state_templates: Sequence[optax.OptState],
) -> tuple[list[PyTree], list[optax.GradientTransformation], list[optax.OptState], State, DictConfig]: ...


@overload
def load_ckpt(
    path: Path,
    *,
    part: Literal["model_state_config"],
    model_templates: Sequence[PyTree],
) -> tuple[list[PyTree], State, DictConfig]: ...


@overload
def load_ckpt(
    path: Path,
    *,
    part: Literal["model"],
    model_templates: Sequence[PyTree],
) -> list[PyTree]: ...


@overload
def load_ckpt(
    path: Path,
    *,
    part: Literal["opt"],
    optimizer_templates: Sequence[optax.GradientTransformation],
) -> list[optax.GradientTransformation]: ...


@overload
def load_ckpt(
    path: Path,
    *,
    part: Literal["opt_state"],
    opt_state_templates: Sequence[optax.OptState],
) -> list[optax.OptState]: ...


@overload
def load_ckpt(path: Path, *, part: Literal["state"]) -> State: ...


@overload
def load_ckpt(path: Path, *, part: Literal["config"]) -> DictConfig: ...


def load_ckpt(
    path: str | Path,
    *,
    part: CheckpointPart = "model",
    model_templates: Sequence[PyTree] | None = None,
    optimizer_templates: Sequence[optax.GradientTransformation] | None = None,
    opt_state_templates: Sequence[optax.OptState] | None = None,
) -> (
    tuple[list[PyTree], list[optax.GradientTransformation], list[optax.OptState], State, DictConfig]
    | tuple[list[PyTree], State, DictConfig]
    | list[PyTree]
    | list[optax.GradientTransformation]
    | list[optax.OptState]
    | State
    | DictConfig
):
    with tarfile.open(path, "r:gz") as tar:

        def get_model() -> list[PyTree]:
            if model_templates is None:
                raise ValueError("model_template must be provided to load model weights")
            models: list[PyTree] = []
            for i, model_template in enumerate(model_templates):
                if (model := tar.extractfile(f"model_{i}")) is None:
                    raise ValueError(f"Checkpoint does not contain a model file: {path}")
                models.append(eqx.tree_deserialise_leaves(io.BytesIO(model.read()), model_template))
            return models

        def get_opt() -> list[optax.GradientTransformation]:
            if optimizer_templates is None:
                raise ValueError("optimizer_template must be provided to load optimizer")
            opts: list[optax.GradientTransformation] = []
            for i, optimizer_template in enumerate(optimizer_templates):
                if (opt := tar.extractfile(f"optimizer_{i}")) is None:
                    raise ValueError(f"Checkpoint does not contain an optimizer file: {path}")
                opts.append(eqx.tree_deserialise_leaves(io.BytesIO(opt.read()), optimizer_template))
            return opts

        def get_opt_state() -> list[optax.OptState]:
            if opt_state_templates is None:
                raise ValueError("opt_state_template must be provided to load optimizer state")
            opt_states: list[optax.OptState] = []
            for i, opt_state_template in enumerate(opt_state_templates):
                if (opt_state := tar.extractfile(f"opt_state_{i}")) is None:
                    raise ValueError(f"Checkpoint does not contain an optimizer state file: {path}")
                opt_states.append(eqx.tree_deserialise_leaves(io.BytesIO(opt_state.read()), opt_state_template))
            return opt_states

        def get_state() -> State:
            if (state := tar.extractfile("state")) is None:
                raise ValueError(f"Checkpoint does not contain a state file: {path}")
            return State.from_dict(**json.loads(state.read().decode()))

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

    def save_checkpoint(
        self,
        models: Sequence[PyTree] | None = None,
        optimizers: Sequence[optax.GradientTransformation] | None = None,
        opt_states: Sequence[optax.OptState] | None = None,
        aux_data: PyTree | None = None,
        state: State | None = None,
    ) -> Path:
        """Save a checkpoint.

        Args:
            models: The models to save
            optimizers: The optimizers to save
            opt_states: The optimizer states to save
            aux_data: Additional data to save
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
        with tarfile.open(ckpt_path, "w:gz") as tar:

            def add_file(name: str, buf: io.BytesIO) -> None:
                tarinfo = tarfile.TarInfo(name)
                tarinfo.size = buf.tell()
                buf.seek(0)
                tar.addfile(tarinfo, buf)

            # Save model using Equinox
            if models is not None:
                for i, model in enumerate(models):
                    with io.BytesIO() as buf:
                        eqx.tree_serialise_leaves(buf, model)
                        add_file(f"model_{i}", buf)

            # Save optimizer using Equinox
            if optimizers is not None:
                for i, optimizer in enumerate(optimizers):
                    with io.BytesIO() as buf:
                        eqx.tree_serialise_leaves(buf, optimizer)
                        add_file(f"optimizer_{i}", buf)

            # Save optimizer state using Equinox
            if opt_states is not None:
                for i, opt_state in enumerate(opt_states):
                    with io.BytesIO() as buf:
                        eqx.tree_serialise_leaves(buf, opt_state)
                        add_file(f"opt_state_{i}", buf)

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
            add_file_bytes("config", OmegaConf.to_yaml(self.config, sort_keys=True).encode())

        # Updates the symlink to the new checkpoint
        last_ckpt_path.unlink(missing_ok=True)
        try:
            last_ckpt_path.symlink_to(ckpt_path.relative_to(last_ckpt_path.parent))
        except FileExistsError:
            logger.exception("Exception while trying to update %s", ckpt_path)

        # Calls the base callback
        self.on_after_checkpoint_save(ckpt_path, state)

        return ckpt_path

    @classmethod
    def load_config(cls, ckpt_path: str | Path) -> Config:
        return cls.get_config(load_ckpt(Path(ckpt_path), part="config"), use_cli=False)

    @classmethod
    def load_task(cls, ckpt_path: str | Path) -> Self:
        return cls(cls.load_config(ckpt_path))
