"""State space models."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def glorot(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0) * jnp.sqrt(2 / sum(shape))


class DiscreteTimeS4(eqx.Module):
    a: Array
    B: Array
    C: Array
    proj_in: eqx.nn.Linear
    proj_out: eqx.nn.Linear

    def __init__(
        self,
        hidden_size: int,
        projection_size: int,
        input_size: int,
        output_size: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.a = glorot(key, (hidden_size,))  # Diagonal vector of square & diagonal matrix A
        self.B = glorot(key, (projection_size, hidden_size))
        self.C = glorot(key, (hidden_size, projection_size))
        self.proj_in = eqx.nn.Linear(input_size, projection_size, key=key)
        self.proj_out = eqx.nn.Linear(projection_size, output_size, key=key)

    def __call__(self, h: Array, x: Array) -> tuple[Array, Array]:
        h = self.a * h + self.B.T @ x
        y = self.C.T @ h
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


class S4(eqx.Module):
    a: Array
    B: Array
    C: Array
    proj_in: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    delta: Array

    def __init__(
        self,
        hidden_size: int,
        projection_size: int,
        input_size: int,
        output_size: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.a = glorot(key, (hidden_size,))  # Diagonal vector of square & diagonal matrix A
        self.B = glorot(key, (projection_size, hidden_size))
        self.C = glorot(key, (hidden_size, projection_size))
        self.proj_in = eqx.nn.Linear(input_size, projection_size, key=key)
        self.proj_out = eqx.nn.Linear(projection_size, output_size, key=key)
        self.delta = jax.random.uniform(key, (hidden_size,))

    def __call__(self, h: Array, x: Array) -> tuple[Array, Array]:
        delta_a = self.delta * self.a
        a_bar = jnp.exp(delta_a)
        B_bar = jnp.linalg.inv(delta_a) * (a_bar - 1) @ (self.delta * self.B)
        h = a_bar * h + B_bar.T @ x
        y = self.C.T @ h
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


class S6(eqx.Module):
    a: Array
    B: Array
    C: Array
    proj_in: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    delta: Array

    def __init__(
        self,
        hidden_size: int,
        projection_size: int,
        input_size: int,
        output_size: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.a = glorot(key, (hidden_size,))
        self.B = glorot(key, (projection_size, hidden_size))
        self.C = glorot(key, (hidden_size, projection_size))
        self.proj_in = eqx.nn.Linear(input_size, projection_size, key=key)
        self.proj_out = eqx.nn.Linear(projection_size, output_size, key=key)
        self.delta = jax.random.uniform(key, (hidden_size,))

    def __call__(self, h: Array, x: Array) -> tuple[Array, Array]:
        h = self.a * h + self.B.T @ x
        y = self.C.T @ h
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
