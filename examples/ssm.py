"""Trains a state space model on a sine wave."""

from dataclasses import dataclass
from typing import Iterator, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray

import xax


def make_sine_dataset(n_samples: int, seq_len: int) -> tuple[Array, Array]:
    """Generate sine sequences and next-step targets."""
    x = jnp.linspace(0, 2 * jnp.pi, seq_len + 1)
    keys = jax.random.split(jax.random.PRNGKey(0), n_samples)
    shifts = jax.vmap(lambda k: jax.random.uniform(k, (), minval=0, maxval=2 * jnp.pi))(keys)
    waves = jnp.sin(x[None, :] + shifts[:, None])
    return waves[:, :-1, None], waves[:, 1:, None]  # (B, T, 1), (B, T, 1)


@dataclass
class Config(xax.Config):
    batch_size: int = xax.field(64)
    learning_rate: float = xax.field(1e-3)
    hidden_size: int = xax.field(32)
    input_size: int = xax.field(1)
    projection_size: int = xax.field(16)
    output_size: int = xax.field(1)
    sequence_length: int = xax.field(50)
    num_samples: int = xax.field(10_000)
    model_type: str = xax.field("s4", help="The model to use")


class RecurrentModel(Protocol):
    def predict_sequence(self, x_seq: Array) -> Array: ...


class RNN(eqx.Module):
    rnn_cell: eqx.nn.GRUCell
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        key: PRNGKeyArray,
    ):
        rnn_key, out_key = jax.random.split(key)
        self.rnn_cell = eqx.nn.GRUCell(input_size=input_size, hidden_size=hidden_size, key=rnn_key)
        self.output_layer = eqx.nn.Linear(hidden_size, output_size, key=out_key)

    def __call__(self, h: Array, x: Array) -> tuple[Array, Array]:
        h = self.rnn_cell(x, h)
        y = self.output_layer(h)
        return h, y

    def predict_sequence(self, x_seq: Array) -> Array:
        h = jnp.zeros(self.rnn_cell.hidden_size)

        def unroll(h: Array, x: Array) -> tuple[Array, Array]:
            h, y = self(h, x)
            return h, y

        _, y_seq = jax.lax.scan(unroll, h, x_seq)
        return y_seq


class SinePrediction(xax.Task[Config]):
    def get_model(self, key: PRNGKeyArray) -> RecurrentModel:
        match self.config.model_type:
            case "rnn":
                return RNN(self.config, key=key)

            case "discrete-s4":
                return xax.DiscreteTimeS4(self.config, key=key)

            case "s4":
                return xax.S4(
                    hidden_size=self.config.hidden_size,
                    projection_size=self.config.projection_size,
                    input_size=self.config.input_size,
                    output_size=self.config.output_size,
                    key=key,
                )

            case _:
                raise ValueError(f"Unknown model type: {self.config.model_type}")

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adam(self.config.learning_rate)

    def get_output(self, model: RecurrentModel, batch: tuple[Array, Array]) -> Array:
        x_batched, _ = batch
        return jax.vmap(model.predict_sequence)(x_batched)

    def compute_loss(self, model: RecurrentModel, batch: tuple[Array, Array], output: Array) -> Array:
        _, y = batch
        return jnp.mean((output - y) ** 2)

    def get_data_iterator(self, phase: xax.Phase) -> Iterator:
        X, Y = make_sine_dataset(self.config.num_samples, self.config.sequence_length)
        split = int(0.8 * self.config.num_samples)
        if phase == "train":
            X, Y = X[:split], Y[:split]
        else:
            X, Y = X[split:], Y[split:]

        key = jax.random.PRNGKey(0)
        while True:
            key, subkey = jax.random.split(key)
            idx = jax.random.randint(subkey, (self.config.batch_size,), 0, X.shape[0])
            yield X[idx], Y[idx]


if __name__ == "__main__":
    # python -m examples.ssm
    SinePrediction.launch(Config())
