# examples/graph_logging.py

import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from xax.task.task import Task, Config
from xax.utils.graph_logging import log_jax_graph


class DummyTask(Task[Config]):
    """Concrete Task so we can trace train_step."""

    def get_model(self, key):
        # simple linear model y = w * x
        @eqx.filter_jit
        class Model(eqx.Module):
            w: jnp.ndarray
            def __call__(self, x):
                return self.w * x
        return Model(w=jnp.array(2.0))

    def get_optimizer(self):
        # basic SGD
        return optax.sgd(learning_rate=0.1)

    def get_output(self, model, batch, state):
        # mean-squared error between model(x) and y
        preds = model(batch["x"])
        return jnp.mean((preds - batch["y"]) ** 2)


def main():
    # 1) Build minimal config & task
    cfg = Config(
        valid_every_n_steps=None,
        valid_every_n_seconds=None,
        max_steps=1,
        batch_size=4,
        random_seed=0,
    )
    task = DummyTask(cfg)

    # 2) Initialize model, optimizer, state
    key = jax.random.PRNGKey(cfg.random_seed)
    models, optimizers, opt_states, state = task.load_initial_state(key, load_optimizer=True)
    model_arr, model_static = eqx.partition(models[0], task.model_partition_fn)
    optimizer = optimizers[0]
    opt_state = opt_states[0]

    # 3) Make a toy batch
    example_batch = {
        "x": jnp.ones((cfg.batch_size,)),
        "y": jnp.ones((cfg.batch_size,)) * 3.0,
    }

    # 4) Prepare logdir
    logdir = os.environ.get("XAX_LOGDIR", "logs/graph")
    os.makedirs(logdir, exist_ok=True)

    # 5) Dump the graph
    log_jax_graph(
        fn=task.train_step,
        example_args=(model_arr, model_static, optimizer, opt_state, example_batch, state),
        logdir=logdir,
        step=0,
    )
    print(f"Written graph to {logdir}.  Run `tensorboard --logdir={logdir}`")


if __name__ == "__main__":
    main()
