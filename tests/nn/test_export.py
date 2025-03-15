from jaxtyping import PyTree
import tensorflow as tf
import flax.linen as nn
import jax
import jax.numpy as jnp
from orbax.export import ExportManager, JaxModule, ServingConfig


# Define a simple Flax module that accepts a dict as input.
class FlaxModule(nn.Module):
    @nn.compact
    def __call__(self, inputs: dict) -> jnp.ndarray:
        x = inputs["features"]
        # A simple Dense layer that returns a JAX array.
        x = nn.Dense(features=10)(x)
        return x


# Instantiate the model and initialize parameters.
model = FlaxModule()
key = jax.random.PRNGKey(0)
dummy_input = {"features": jnp.ones((1, 28))}
params = model.init(key, dummy_input)


def forward_fn(params: PyTree, inputs: dict) -> jnp.ndarray:
    return model.apply(params, inputs)


# Create an input signature.
# Since our function expects a single argument (a dict), we use a list with one element.
# Each key in the dict should map to a tf.TensorSpec that defines the expected shape and dtype.
input_signature = [{"features": tf.TensorSpec(shape=[None, 28], dtype=tf.float32)}]


# Define a preprocess function.
# This function converts the nested tf.Tensor structure into numpy arrays.
# It returns a tuple, where the first element corresponds to the first argument of forward_fn (after the stored params).
def preprocess_fn(inputs: dict) -> tuple[dict[str, jnp.ndarray]]:
    # Convert each tf.Tensor in the dict to a numpy array.
    np_inputs = tf.nest.map_structure(lambda t: t.numpy(), inputs)
    # Return a tuple with a single element: the dict of numpy arrays.
    return (np_inputs,)


# Define a postprocess function.
# It wraps the raw model output (a JAX array) into a dictionary.
def postprocess_fn(output: jnp.ndarray) -> dict[str, jnp.ndarray]:
    return {"output": output}


# Create the JaxModule; note that we pass the entire Flax model parameters as the "params"
jax_module = JaxModule(params, forward_fn)

# Create the ExportManager with the ServingConfig.
export_manager = ExportManager(
    jax_module,
    [
        ServingConfig(
            "serving_default",
            input_signature=input_signature,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
        )
    ],
)

# Export the model to the given directory.
export_dir = "path/to/export_dir"
export_manager.save(export_dir)
