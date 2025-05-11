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
        self.optimizer = None
        self.state = None

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def train_step(self, model, optimizer, state, batch):
        loss = 0.0
        grads = None
        return loss, grads, state

    def load_initial_state(self, key, load_optimizer=False):
        return [None], [None], [None], None


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
    opt_state = opt_states[0]

    # Define a fixed input shape for jax.jit
    example_batch = {
        "x": jnp.ones((cfg.batch_size, 4)),  # Input shape is (batch_size, 4)
        "y": jnp.ones((cfg.batch_size,)) * 3.0,
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
