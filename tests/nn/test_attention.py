"""Tests for attention mechanisms."""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

import xax


@pytest.mark.parametrize("use_rotary_embeddings", [True, False])
def test_self_attention_block_loopback(use_rotary_embeddings: bool) -> None:
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    block = xax.SelfAttentionBlock(
        embed_dim=32,
        num_heads=2,
        key=subkey,
        context_length=5,
        use_rotary_embeddings=use_rotary_embeddings,
    )

    def scan_fn(
        carry: tuple[Array, xax.AttentionCache],
        _: Array,
    ) -> tuple[tuple[Array, xax.AttentionCache], Array]:
        x, cache = carry
        x, cache = block.forward(x, cache=cache)
        return (x, cache), x

    # Gets a random starting vector.
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (1, block.embed_dim))

    # Autoregressive unrolling.
    cache = block.init_cache(x.dtype)
    _, xs = xax.scan(scan_fn, (x, cache), length=10, jit_level=-1)
    xs = xs.squeeze(1)
    prev_xs = jnp.concatenate([x, xs[:-1]], axis=0)

    # Calls the batched forward function.
    mask = block.init_mask(10, with_cache=True)
    next_xs, _ = block.forward(prev_xs, mask=mask, cache=cache)

    assert jnp.allclose(xs, next_xs, atol=1e-6)


@pytest.mark.parametrize("use_rotary_embeddings", [True, False])
def test_transformer_block_loopback(use_rotary_embeddings: bool) -> None:
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    block = xax.TransformerBlock(
        embed_dim=32,
        num_heads=2,
        ff_dim=64,
        key=subkey,
        cross_attention=True,
        context_length=5,
        use_rotary_embeddings=use_rotary_embeddings,
    )

    def scan_fn(
        carry: tuple[Array, xax.AttentionCacheDict],
        _: Array,
    ) -> tuple[tuple[Array, xax.AttentionCacheDict], Array]:
        x, cache = carry
        x, cache = block.forward(x, cache=cache)
        return (x, cache), x

    # Gets a random starting vector.
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (1, block.embed_dim))

    # Gets a random context vector.
    key, subkey = jax.random.split(key)
    context_sn = jax.random.normal(subkey, (3, block.embed_dim))

    # Autoregressive unrolling.
    cache = block.init_cache(x.dtype, context_sn=context_sn)
    _, xs = xax.scan(scan_fn, (x, cache), length=10)
    xs = xs.squeeze(1)
    prev_xs = jnp.concatenate([x, xs[:-1]], axis=0)

    # Calls the batched forward function.
    mask = block.init_mask(10, with_cache=True)
    next_xs, _ = block.forward(prev_xs, context_sn=context_sn, self_mask=mask, cache=cache)

    assert jnp.allclose(xs, next_xs, atol=1e-6)


@pytest.mark.parametrize("use_rotary_embeddings", [True, False])
def test_transformer_stack_loopback(use_rotary_embeddings: bool) -> None:
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    stack = xax.TransformerStack(
        embed_dim=32,
        num_heads=2,
        ff_dim=64,
        num_layers=3,
        key=subkey,
        cross_attention=True,
        context_length=5,
        use_rotary_embeddings=use_rotary_embeddings,
    )

    def scan_fn(
        carry: tuple[Array, xax.TransformerCache],
        _: Array,
    ) -> tuple[tuple[Array, xax.TransformerCache], Array]:
        x, cache = carry
        x, cache = stack.forward(x, cache=cache)
        return (x, cache), x

    # Gets a random starting vector.
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (1, stack.layers[0].embed_dim))

    # Gets a random context vector.
    key, subkey = jax.random.split(key)
    context_sn = jax.random.normal(subkey, (3, stack.layers[0].embed_dim))

    # Autoregressive unrolling.
    cache = stack.init_cache(x.dtype, x_tn=context_sn)
    _, xs = xax.scan(scan_fn, (x, cache), length=10)
    xs = xs.squeeze(1)
    prev_xs = jnp.concatenate([x, xs[:-1]], axis=0)

    # Calls the batched forward function.
    mask = stack.init_mask(10)
    next_xs, _ = stack.forward(prev_xs, context_sn=context_sn, self_mask=mask, cache=cache)

    assert jnp.allclose(xs, next_xs, atol=1e-6)


@pytest.mark.parametrize("use_rotary_embeddings", [True, False])
def test_transformer_loopback(use_rotary_embeddings: bool) -> None:
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    transformer = xax.Transformer(
        vocab_size=1000,
        embed_dim=32,
        num_heads=2,
        ff_dim=64,
        num_layers=2,
        key=subkey,
        cross_attention=False,
        context_length=5,
        use_rotary_embeddings=use_rotary_embeddings,
    )

    # Generates a random sequence.
    key, subkey = jax.random.split(key)
    x = jax.random.randint(subkey, (5,), 0, 1000)
    seq = transformer.generate_sequence(x, max_len=10, top_k=1)

    # Does next token prediction using the generated sequence.
    cache = transformer.init_cache(x.dtype)
    _, cache = transformer.encode(x, cache=cache)
    mask = transformer.init_mask(10, with_cache=True)
    next_x, _ = transformer.forward(seq[4:-1], mask=mask, cache=cache)
    pseq = next_x.argmax(axis=-1)

    assert jnp.allclose(pseq, seq[5:])
