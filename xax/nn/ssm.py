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



