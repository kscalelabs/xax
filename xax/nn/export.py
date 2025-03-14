from typing import Any, Callable, Protocol

import jax
import numpy as np
from jaxtyping import Array, PyTree

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("TensorFlow is required to export the model - install with `pip install xax[export]`")

try:
    from orbax.export import ExportManager, JaxModule, ServingConfig
    from orbax.export.constants import ExportModelType
except ImportError:
    raise ImportError("Orbax is required to export the model - install with `pip install xax[export]`")


def create_serving_config(model: JaxModule, variables: dict[str, Any]) -> ServingConfig:
    """Create a serving config for the model.

    Args:
        model: The model to serve.
        variables: The model variables.

    Returns:
        A ServingConfig for the model.
    """
    # Get a dummy observation and command to determine input shapes
    dummy_obs: dict[str, np.ndarray] = {}
    dummy_cmd: dict[str, np.ndarray] = {}

    # Extract normalization parameters to determine expected input shapes
    for key in variables["normalization"]:
        if key.startswith("obs_mean_"):
            obs_name = key[len("obs_mean_") :]
            shape = variables["normalization"][key].shape
            # Create dummy tensor with appropriate shape
            dummy_obs[obs_name] = np.zeros((1,) + shape)

    # Add dummy commands - this is a simplification, you may need to adjust based on your model
    dummy_cmd["linear_velocity_command"] = np.zeros((1, 2))
    dummy_cmd["angular_velocity_command"] = np.zeros((1, 1))

    # Create TensorSpec for inputs
    obs_signature = {k: tf.TensorSpec(v.shape, tf.float32, name=k) for k, v in dummy_obs.items()}
    cmd_signature = {k: tf.TensorSpec(v.shape, tf.float32, name=k) for k, v in dummy_cmd.items()}

    # Define a preprocessor function to pack obs and cmd into a single input
    def preprocessor(obs: dict[str, tf.Tensor], cmd: dict[str, tf.Tensor]) -> Any:
        # Convert TensorFlow tensors to NumPy arrays
        obs_dict = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in obs.items()}
        cmd_dict = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in cmd.items()}

        # Pack into a tuple for the model function
        return (obs_dict, cmd_dict)

    return ServingConfig(
        signature_key="serve_actor",
        input_signature=[obs_signature, cmd_signature],
        tf_preprocessor=preprocessor,
        method_key="jax_module_default_method",
    )


def export_actor_model(jax_module: JaxModule, variables: dict[str, Any], export_dir: str) -> None:
    """Export only the actor component of a trained ActorCriticAgent model.

    This function exports only the actor part of the model, but preserves the normalization
    functionality which is crucial for correct inference.

    Args:
        jax_module: The model to export.
        variables: The model variables including normalization parameters.
        export_dir: Directory to save the exported model.
    """
    # Create serving config
    serving_config = create_serving_config(jax_module, variables)

    # Create export manager
    export_manager = ExportManager(module=jax_module, serving_configs=[serving_config])

    export_manager.save(export_dir)
    print(f"Actor model exported to {export_dir}")


class TaskWithModelAndCheckpoint(Protocol):
    def load_checkpoint(self, checkpoint_path: str, part: str) -> dict[str, Any]: ...
    def get_model(self, key: jax.Array) -> JaxModule: ...


def export_actor_from_task(task: TaskWithModelAndCheckpoint, checkpoint_path: str, export_dir: str) -> None:
    """Export only the actor component from a task's checkpoint.

    This is a helper function to export just the actor model from a task.
    It loads the checkpoint, extracts the model and variables, and exports only the actor.

    Args:
        task: The task instance that contains the model
        checkpoint_path: Path to the checkpoint to load
        export_dir: Directory to save the exported model
    """
    variables = task.load_checkpoint(checkpoint_path, part="model")
    model = task.get_model(jax.random.PRNGKey(0))
    export_actor_model(model, variables, export_dir)

    print(f"Actor model exported from checkpoint {checkpoint_path} to {export_dir}")
