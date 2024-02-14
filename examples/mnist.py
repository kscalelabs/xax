"""Trains a simple convolutional neural network on the MNIST dataset.

Run this example with `python -m examples.mnist`.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from dpshdl.impl.mnist import MNIST
from jaxtyping import Array, Float, Int

import xax

Xb = Float[Array, "batch 1 28 28"]
X = Float[Array, "1 28 28"]

Yb = Int[Array, "batch"]
Y = Int[Array, ""]

Yhatb = Float[Array, "batch 10"]
Yhat = Float[Array, "10"]

Batch = tuple[Xb, Yb]
Loss = Float[Array, ""]


class Model(eqx.Module):
    layers: list

    def __init__(self, rng_key: Array) -> None:
        super().__init__()

        # Split the PRNG key into four keys for the four layers.
        key1, key2, key3, key4 = jax.random.split(rng_key, 4)

        self.layers = [
            eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(1728, 512, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, 64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(64, 10, key=key4),
            jax.nn.log_softmax,
        ]

    def forward(self, x: X) -> Y:
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x: Xb) -> Yb:
        return jax.vmap(self.forward)(x)


def cross_entropy(y: Yb, pred_y: Yhatb) -> Loss:
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


@dataclass
class Config(xax.Config):
    in_dim: int = xax.field(1, help="Number of input dimensions")


class MnistClassification(xax.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def get_model(self) -> eqx.Module:
        return Model(self.prng_key)

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adam(1e-3)

    def get_output(self, model: eqx.Module, batch: Batch, state: xax.State) -> Yhatb:
        x, _ = batch
        y_hat = model(x[:, None])
        return y_hat

    def compute_loss(self, model: eqx.Module, batch: Batch, output: Yhatb, state: xax.State) -> Array:
        (_, y), y_hat = batch, output
        return cross_entropy(y, y_hat)

    def get_dataset(self, phase: xax.Phase) -> MNIST:
        return MNIST(
            train=phase == "train",
            root_dir=xax.get_data_dir() / "mnist",
            dtype="float32",
        )


def test_dataloader_adhoc() -> None:
    task = MnistClassification.get_task(Config(batch_size=16, num_dataloader_workers=0))
    pf = task.get_prefetcher(task.get_dataloader(task.get_dataset("train"), "train"))
    pf.test(max_samples=1000)


if __name__ == "__main__":
    # python -m examples.mnist
    config = Config(batch_size=16)
    config.train_dl.num_workers = 0
    MnistClassification.launch(config)
