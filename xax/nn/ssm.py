"""State space models."""

from abc import ABC, abstractmethod
from typing import Literal

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
        self.a = jax.nn.initializers.glorot_uniform()(key, (hidden_size,))
        self.B = jax.nn.initializers.glorot_uniform()(key, (projection_size, hidden_size))
        self.C = jax.nn.initializers.glorot_uniform()(key, (hidden_size, projection_size))
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


class S4Layer(eqx.Module):
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
        self.a = jax.nn.initializers.glorot_uniform()(key, (hidden_size,))
        self.B = jax.nn.initializers.glorot_uniform()(key, (projection_size, hidden_size))
        self.C = jax.nn.initializers.glorot_uniform()(key, (hidden_size, projection_size))
        self.proj_in = eqx.nn.Linear(input_size, projection_size, key=key)
        self.proj_out = eqx.nn.Linear(projection_size, output_size, key=key)
        self.delta = jax.random.uniform(key, (hidden_size,))

    def __call__(self, h: Array, x: Array) -> tuple[Array, Array]:
        delta_a = self.delta * self.a
        a_bar = jnp.exp(delta_a)
        b_bar = jnp.linalg.inv(delta_a) * (a_bar - 1) @ (self.delta * self.B)
        h = a_bar * h + b_bar.T @ x
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


class S6Layer(eqx.Module):
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
        self.a = jax.nn.initializers.glorot_uniform()(key, (hidden_size,))
        self.B = jax.nn.initializers.glorot_uniform()(key, (projection_size, hidden_size))
        self.C = jax.nn.initializers.glorot_uniform()(key, (hidden_size, projection_size))
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


class BaseSSMBlock(eqx.Module, ABC):
    @abstractmethod
    def forward(self, h: Array, x: Array) -> Array:
        pass


class SSMBlock(BaseSSMBlock):
    a_mat: Array
    b_mat: Array

    def __init__(self, hidden_size: int, *, key: PRNGKeyArray) -> None:
        key_a, key_b = jax.random.split(key)
        self.a_mat = glorot(key_a, (hidden_size, hidden_size))
        self.b_mat = glorot(key_b, (hidden_size, hidden_size))

    def forward(self, h: Array, x: Array) -> Array:
        h = self.a_mat @ h + self.b_mat.T @ x
        return h

    def get_kernel(self, length: int) -> Array:
        return self.a_mat


class DiagSSMBlock(BaseSSMBlock):
    a_mat: Array
    b_mat: Array

    def __init__(self, hidden_size: int, *, key: PRNGKeyArray) -> None:
        keys = jax.random.split(key, 2)
        self.a_mat = glorot(keys[0], (hidden_size,))
        self.b_mat = glorot(keys[1], (hidden_size, hidden_size))

    def forward(self, h: Array, x: Array) -> Array:
        h = self.a_mat * h + self.b_mat.T @ x
        h = jax.nn.tanh(h)
        return h

    def get_kernel(self, length: int) -> Array:
        """Returns the kernel with time as the final dimension."""
        exponents = jnp.arange(length)
        kernel = jnp.power(self.a_mat[:, None], exponents)  # (H, L)
        kernel = kernel[:, None, :]  # (H, 1, L)
        return kernel

    def forward_across_time(self, x: Array) -> Array:
        """Convolves x (T, H) across time using the kernel."""
        tsz, nhid = x.shape

        # Compute s = x @ U.T + b, with shape (N, T, H)
        s = self.b_mat.T @ x
        s = s.T  # (H, T)

        kernel = self.get_kernel(tsz)  # (H, 1, T)
        kernel_flipped = jnp.flip(kernel, axis=-1)

        # Pad s on the left along the time axis (pad length T-1)
        s_padded = jnp.pad(s, ((0, 0), (0, 0), (tsz - 1, 0)))

        # Perform depthwise (grouped) 1D convolution.
        # We use input shape (N, H, L) and kernel shape (H, 1, T) with feature_group_count=H.
        # The dimension_numbers are chosen so that the channel dimension is second.
        conv_out = jax.lax.conv_general_dilated(
            s_padded,
            kernel_flipped,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCH", "OIH", "NCH"),
            feature_group_count=nhid,
        )
        # conv_out has shape (N, H, T); transpose to (N, T, H)
        conv_out = jnp.transpose(conv_out, (0, 2, 1))
        return conv_out

    def naive_forward_accross_time(self, x: Array) -> Array:
        """Naively forward across time."""

        def step(h: Array, x: Array) -> tuple[Array, Array]:
            h = self.forward(h, x)
            return h, h

        h_0 = jnp.zeros(self.a_mat.shape[0])
        _, h_seq = jax.lax.scan(step, h_0, x)
        return h_seq


class S4(eqx.Module):
    vocab_embedding: eqx.nn.Embedding
    proj_in: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    blocks: list[BaseSSMBlock]
    num_layers: int = eqx.static_field()
    hidden_size: int = eqx.static_field()
    skip_connections: bool = eqx.static_field()

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        block_type: Literal["ssm", "diag"] = "ssm",
        skip_connections: bool = False,
        *,
        key: PRNGKeyArray,
    ) -> None:
        vocab_key, s4_key = jax.random.split(key, 2)
        self.vocab_embedding = eqx.nn.Embedding(input_size, hidden_size, key=vocab_key)
        self.proj_in = eqx.nn.Linear(hidden_size, hidden_size, key=key)
        self.proj_out = eqx.nn.Linear(hidden_size, output_size, key=key)

        block_keys = jax.random.split(s4_key, num_layers)

        def get_block(key: PRNGKeyArray) -> BaseSSMBlock:
            match block_type:
                case "ssm":
                    return SSMBlock(hidden_size, key=key)
                case "diag":
                    return DiagSSMBlock(hidden_size, key=key)
                case _:
                    raise ValueError(f"Unknown block type: {block_type}")

        self.blocks = [get_block(block_keys[i]) for i in range(num_layers)]
        self.skip_connections = skip_connections
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def __call__(self, hs: list[Array], x: Array) -> tuple[list[Array], Array]:
        new_hs = []
        for i, block in enumerate(self.blocks):
            h = block.forward(hs[i], x)
            new_hs.append(h)
            xh = jax.nn.gelu(h)
            x = xh + x if self.skip_connections else xh
        y = self.proj_out(x)
        return new_hs, y

    def _embed_input(self, x: Array) -> Array:
        """U is the input to the S4 cell."""
        embedded = self.vocab_embedding(x)
        return jax.nn.gelu(self.proj_in(embedded))

    def predict_sequence(self, x_seq: Array) -> Array:
        x_emb = jax.vmap(self._embed_input)(x_seq)
        hs = [jnp.zeros(self.hidden_size) for _ in range(self.num_layers)]

        def step(hs: list[Array], x: Array) -> tuple[list[Array], Array]:
            hs, y = self(hs, x)
            return hs, y

        _, y_seq = jax.lax.scan(step, hs, x_emb)
        return y_seq
