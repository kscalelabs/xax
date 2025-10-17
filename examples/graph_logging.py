import os
import jax
import jax.numpy as jnp
import equinox as eqx
from xax.task.task import Task, Config
from xax.task.loggers.tensorboard import TensorboardLogger  # Correct import for logging
from xax.utils.tensorboard import TensorboardWriter  # Use TensorboardWriter for actual logging
from tensorboard.compat.proto.graph_pb2 import GraphDef
from xax.utils.graph_logging import log_jax_graph
import optax
import re

class MyModel(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self):
        self.weight = jnp.ones((4, 3))  # Weight matrix shape should match input x shape
        self.bias = jnp.zeros(3)  # Bias vector with 3 elements

    def __call__(self, x):
        return jnp.dot(x, self.weight) + self.bias  # Dot product and adding bias
    

class MyTask(Task):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.model = MyModel()  # Initialize the model
        self.optimizer = optax.adam(learning_rate=1e-3)  # Adam optimizer
        self.state = None

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer
    
    def get_output(self, model: eqx.Module, batch: dict, state: dict) -> jnp.ndarray:
        """
        Override the `get_output` method to calculate the output of the model.
        In this case, we compute the model's prediction based on the input batch.

        Args:
            model: The model to get output from.
            batch: The batch containing the input data.
            state: The training state (if needed).

        Returns:
            The output of the model (in this case, the predictions).
        """
        # Perform forward pass to get output from the model
        if model is None:
            raise ValueError("Model is None when trying to compute the output")
        return model(batch["x"])
    
    def train_step(self, model_arr, model_static, optimizer, opt_state, batch, state):
        """
        Perform one training step: compute output, loss, gradients, and update model.

        Args:
            model_arr: The current model parameters.
            model_static: Static model parameters.
            optimizer: The optimizer used for updating the model.
            opt_state: The current optimizer state.
            batch: The current batch of data.
            state: Additional state for training.

        Returns:
            model_arr: Updated model parameters.
            opt_state: Updated optimizer state.
            output: Model output for the current batch.
            metrics: Computed metrics (loss, etc.).
        """
        output = self.get_output(model_arr, batch, state)
        loss = jnp.mean((output - batch["y"]) ** 2)  # MSE loss
        grads = jax.grad(lambda m: jnp.mean((self.get_output(m, batch, state) - batch["y"]) ** 2))(model_arr)
        updates, opt_state = optimizer.update(grads, opt_state)
        model_arr = optax.apply_updates(model_arr, updates)  # Apply optimizer updates
        metrics = {"loss": loss}  # You can add more metrics here
        return model_arr, opt_state, output, metrics
    

    def load_initial_state(self, key, load_optimizer=False):
        return [self.model], [self.optimizer], [None], None


def main():
    cfg = Config(
        valid_every_n_steps=10,
        valid_every_n_seconds=None,
        max_steps=100,
        batch_size=4,
        random_seed=0
    )

    task = MyTask(cfg)

    # 1) Prepare logdir for TensorBoard
    logdir = os.environ.get("XAX_LOGDIR", "logs/graph")
    os.makedirs(logdir, exist_ok=True)

    key = jax.random.PRNGKey(cfg.random_seed)
    models, optimizers, opt_states, state = task.load_initial_state(key, load_optimizer=True)
    model_arr, model_static = eqx.partition(models[0], task.model_partition_fn)
    optimizer = optimizers[0]
    #opt_state = opt_states[0]
    opt_state = optimizer.init(model_arr)  # Initialize the optimizer state here
    # Define a fixed input shape for jax.jit
    example_batch = {
        "x": jnp.ones((cfg.batch_size, 4)),  # Input shape is (batch_size, 4)
        "y": jnp.ones((cfg.batch_size, 3)) * 3.0,
    }

    log_jax_graph(
        fn=task.train_step,
        example_args=(model_arr, model_static, optimizer, opt_state, example_batch, state),
        task=task,
        cfg=cfg,
        logdir=logdir,
        step=0,
    )

    print(f"Finished training. Run `tensorboard --logdir={logdir}` to visualize.")


if __name__ == "__main__":
    main()
