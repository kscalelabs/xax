"""State space models."""

from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def glorot(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0) * jnp.sqrt(2 / sum(shape))


class BaseSSMBlock(eqx.Module, ABC):
    @abstractmethod
    def forward(self, h: Array, x: Array) -> Array:
        pass

    @abstractmethod
    def forward_sequence(self, x_seq: Array) -> Array:
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

    def forward_sequence(self, x_seq: Array) -> Array:
        raise NotImplementedError("SSMBlock does not support forward_sequence")


class DiagSSMBlock(BaseSSMBlock):
    a_diag: Array
    b_mat: Array

    def __init__(self, hidden_size: int, *, key: PRNGKeyArray) -> None:
        keys = jax.random.split(key, 2)
        self.a_diag = glorot(keys[0], (hidden_size,))
        self.b_mat = glorot(keys[1], (hidden_size, hidden_size))

    def forward(self, h: Array, x: Array) -> Array:
        h = self.a_diag * h + self.b_mat.T @ x
        # h = jax.nn.tanh(h)
        return h

    def get_kernel(self, length: int) -> Array:
        """Returns the kernel with time as the final dimension."""
        exponents = jnp.arange(length)
        kernel = jnp.power(self.a_diag[:, None], exponents)  # (H, L)
        kernel = kernel[:, None, :]  # (H, 1, L)
        return kernel

    def forward_sequence(self, x_seq: Array) -> Array:
        """Convolves x (T, H) across time using the kernel."""
        seq_len, hidden_size = x_seq.shape

        s = self.b_mat.T @ x_seq.T  # (H, T)
        s_padded = jnp.pad(s, ((0, 0), (seq_len - 1, 0)))[None, :, :]  # (1, H, 2T-1)

        kernel = self.get_kernel(seq_len)  # (H, 1, T)
        kernel_flipped = jnp.flip(kernel, axis=-1)

        conv_out = jax.lax.conv_general_dilated(
            s_padded,
            kernel_flipped,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCT", "OIT", "NCT"),
            feature_group_count=hidden_size,
        )
        conv_out = conv_out[0].T  # (T, H)
        return conv_out

    def naive_forward_sequence(self, x: Array) -> Array:
        """Naively forward across time."""

        def step(h: Array, x: Array) -> tuple[Array, Array]:
            h = self.forward(h, x)
            return h, h

        h_0 = jnp.zeros(self.a_diag.shape[0])
        _, h_seq = jax.lax.scan(step, h_0, x)
        return h_seq



