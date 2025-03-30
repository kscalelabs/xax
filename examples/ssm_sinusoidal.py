from abc import ABC, abstractmethod
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax

from jaxtyping import Array, PRNGKeyArray
from dataclasses import dataclass
from typing import Iterator


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

class RecurrentModel(ABC):
    @abstractmethod
    def predict_sequence(self, x_seq: Array) -> Array:
        pass

class RNN(eqx.Module, RecurrentModel):
    rnn_cell: eqx.nn.GRUCell
    output_layer: eqx.nn.Linear

    def __init__(self, config: Config, *, key: PRNGKeyArray):
        rnn_key, out_key = jax.random.split(key)
        self.rnn_cell = eqx.nn.GRUCell(input_size=config.input_size, hidden_size=config.hidden_size, key=rnn_key)
        self.output_layer = eqx.nn.Linear(config.hidden_size, config.output_size, key=out_key)

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

class DiscreteTimeS4(eqx.Module, RecurrentModel):
    a: Array
    B: Array
    C: Array
    proj_in: eqx.nn.Linear
    proj_out: eqx.nn.Linear

    def __init__(self, config: Config, *, key: PRNGKeyArray):
        glorot = lambda k, shape: jax.random.uniform(k, shape, minval=-1.0, maxval=1.0) * jnp.sqrt(2 / sum(shape))
        self.a = glorot(key, (config.hidden_size,)) # diagonal vector of square & diagonal matrix A
        self.B = glorot(key, (config.projection_size, config.hidden_size))
        self.C = glorot(key, (config.hidden_size, config.projection_size))
        self.proj_in = eqx.nn.Linear(config.input_size, config.projection_size, key=key)
        self.proj_out = eqx.nn.Linear(config.projection_size, config.output_size, key=key)

    def __call__(self, h: Array, x: Array) -> tuple[Array, Array]:
        h = self.a * h + self.B.T @ x
        y = self.C.T @ h # TODO: try adding D to see if it helps?
        return h, y

    def predict_sequence(self, x_seq: Array) -> Array:
        x_proj = jax.vmap(lambda x: jax.nn.relu(self.proj_in(x)))(x_seq)
        h = jnp.zeros(self.a.shape[0])
        
        def scan_fn(h: Array, x: Array) -> tuple[Array, Array]:
            h = self.a * h + self.B.T @ x
            y = self.C.T @ h
            return h, y
        
        _, y_seq = jax.lax.scan(scan_fn, h, x_proj)
        y_out = jax.vmap(self.proj_out)(y_seq)
        return y_out

class S4(eqx.Module, RecurrentModel):
    a: Array
    B: Array
    C: Array
    proj_in: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    delta: Array

    def __init__(self, config: Config, *, key: PRNGKeyArray):
        glorot = lambda k, shape: jax.random.uniform(k, shape, minval=-1.0, maxval=1.0) * jnp.sqrt(2 / sum(shape))
        self.a = glorot(key, (config.hidden_size,)) # diagonal vector of square & diagonal matrix A
        self.B = glorot(key, (config.projection_size, config.hidden_size))
        self.C = glorot(key, (config.hidden_size, config.projection_size))
        self.proj_in = eqx.nn.Linear(config.input_size, config.projection_size, key=key)
        self.proj_out = eqx.nn.Linear(config.projection_size, config.output_size, key=key)
        self.delta = jax.random.uniform(key, (config.hidden_size,))

    def __call__(self, h: Array, x: Array) -> tuple[Array, Array]:
        delta_a = self.delta * self.a
        a_bar = jnp.exp(delta_a)
        B_bar = jnp.linalg.inv(delta_a) * (a_bar - 1) @ (self.delta * self.B)
        h = a_bar * h + B_bar.T @ x
        y = self.C.T @ h # TODO: try adding D to see if it helps?
        return h, y

    def predict_sequence(self, x_seq: Array) -> Array:
        x_proj = jax.vmap(lambda x: jax.nn.gelu(self.proj_in(x)))(x_seq)
        h = jnp.zeros(self.a.shape[0])
        
        def scan_fn(h: Array, x: Array) -> tuple[Array, Array]:
            h = self.a * h + self.B.T @ x
            y = self.C.T @ h
            return h, y
        
        _, y_seq = jax.lax.scan(scan_fn, h, x_proj)
        y_out = jax.vmap(self.proj_out)(y_seq)
        return y_out

class S6(eqx.Module, RecurrentModel):
    a: Array
    B: Array
    C: Array
    proj_in: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    delta: Array

    def __init__(self, config: Config, *, key: PRNGKeyArray):
        glorot = lambda k, shape: jax.random.uniform(k, shape, minval=-1.0, maxval=1.0) * jnp.sqrt(2 / sum(shape))
        self.a = glorot(key, (config.hidden_size,)) # diagonal vector of square & diagonal matrix A
        self.B = glorot(key, (config.projection_size, config.hidden_size))
        self.C = glorot(key, (config.hidden_size, config.projection_size))
        self.proj_in = eqx.nn.Linear(config.input_size, config.projection_size, key=key)
        self.proj_out = eqx.nn.Linear(config.projection_size, config.output_size, key=key)

    def __call__(self, h: Array, x: Array) -> tuple[Array, Array]:
        h = self.a * h + self.B.T @ x
        y = self.C.T @ h # TODO: try adding D to see if it helps?
        return h, y

    def predict_sequence(self, x_seq: Array) -> Array:
        x_proj = jax.vmap(lambda x: jax.nn.gelu(self.proj_in(x)))(x_seq)
        h = jnp.zeros(self.a.shape[0])
        
        def scan_fn(h: Array, x: Array) -> tuple[Array, Array]:
            h = self.a * h + self.B.T @ x
            y = self.C.T @ h
            return h, y
        
        _, y_seq = jax.lax.scan(scan_fn, h, x_proj)
        y_out = jax.vmap(self.proj_out)(y_seq)
        return y_out

class SinePrediction(xax.Task[Config]):
    def get_model(self, key: PRNGKeyArray) -> RecurrentModel:
        # return RNN(self.config, key=key)
        # return DiscreteTimeS4(self.config, key=key)
        return S4(self.config, key=key)

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adam(self.config.learning_rate)

    def get_output(self, model: RecurrentModel, batch: tuple[Array, Array]) -> Array:
        x_batched, _ = batch
        return jax.vmap(model.predict_sequence)(x_batched)

    def compute_loss(self, model: RecurrentModel, batch: tuple[Array, Array], output: Array) -> Array:
        _, y = batch
        return jnp.mean((output - y) ** 2)

    def log_train_step(self, model: RecurrentModel, batch: tuple[Array, Array], output: Array, state: xax.State) -> None:
        _, y = batch
        mse = jnp.mean((output - y) ** 2)
        self.logger.log_scalar("train_mse", mse)

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
    SinePrediction.launch(Config())
