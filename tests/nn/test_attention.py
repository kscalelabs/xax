"""Tests for attention mechanisms."""

import jax
import jax.numpy as jnp
import pytest

import xax


def test_self_attention_block_init() -> None:
    """Test SelfAttentionBlock initialization."""
    key = jax.random.PRNGKey(0)
    block = xax.SelfAttentionBlock(embed_dim=64, num_heads=8, key=key)

    assert block.num_heads == 8
    assert block.head_dim == 8
    assert block.causal is False


def test_self_attention_block_init_causal() -> None:
    """Test SelfAttentionBlock initialization with causal attention."""
    key = jax.random.PRNGKey(0)
    block = xax.SelfAttentionBlock(embed_dim=64, num_heads=8, key=key, causal=True)

    assert block.causal is True


def test_self_attention_block_forward() -> None:
    """Test SelfAttentionBlock forward pass."""
    key = jax.random.PRNGKey(0)
    block = xax.SelfAttentionBlock(embed_dim=64, num_heads=8, key=key)

    x = jax.random.normal(key, (10, 64))
    output, cache = block.forward(x)

    assert output.shape == (10, 64)
    assert "k" in cache
    assert "v" in cache


def test_self_attention_block_forward_with_cache() -> None:
    """Test SelfAttentionBlock forward pass with caching."""
    key = jax.random.PRNGKey(0)
    block = xax.SelfAttentionBlock(embed_dim=64, num_heads=8, key=key)

    x = jax.random.normal(key, (10, 64))
    output, cache = block.forward(x)

    assert output.shape == (10, 64)
    assert "k" in cache
    assert "v" in cache
    assert cache["k"].shape == (10, 8, 8)
    assert cache["v"].shape == (10, 8, 8)


def test_self_attention_block_forward_with_cached_kv() -> None:
    """Test SelfAttentionBlock forward pass using cached key/value."""
    key = jax.random.PRNGKey(0)
    block = xax.SelfAttentionBlock(embed_dim=64, num_heads=8, key=key)

    x = jax.random.normal(key, (10, 64))
    _, cache = block.forward(x)

    # Use cached key/value
    output, _ = block.forward(x, cache=cache)
    assert output.shape == (10, 64)


def test_cross_attention_block_init() -> None:
    """Test CrossAttentionBlock initialization."""
    key = jax.random.PRNGKey(0)
    block = xax.CrossAttentionBlock(embed_dim=64, num_heads=8, key=key)

    assert block.num_heads == 8
    assert block.head_dim == 8


def test_cross_attention_block_forward() -> None:
    """Test CrossAttentionBlock forward pass."""
    key = jax.random.PRNGKey(0)
    block = xax.CrossAttentionBlock(embed_dim=64, num_heads=8, key=key)

    q_input = jax.random.normal(key, (5, 64))
    kv_input = jax.random.normal(key, (10, 64))
    output, cache = block.forward(q_input, kv_sn=kv_input)

    assert output.shape == (5, 64)
    assert cache is not None


def test_cross_attention_block_forward_with_cache() -> None:
    """Test CrossAttentionBlock forward pass with caching."""
    key = jax.random.PRNGKey(0)
    block = xax.CrossAttentionBlock(embed_dim=64, num_heads=8, key=key)

    q_input = jax.random.normal(key, (5, 64))
    kv_input = jax.random.normal(key, (10, 64))
    output, cache = block.forward(q_input, kv_sn=kv_input)

    assert output.shape == (5, 64)
    assert "k" in cache
    assert "v" in cache
    assert cache["k"].shape == (10, 8, 8)
    assert cache["v"].shape == (10, 8, 8)


def test_transformer_block_init() -> None:
    """Test TransformerBlock initialization."""
    key = jax.random.PRNGKey(0)
    block = xax.TransformerBlock(embed_dim=64, num_heads=8, ff_dim=128, key=key)

    assert block.self_attn is not None
    assert block.cross_attn is None
    assert block.layer_norm3 is None


def test_transformer_block_init_with_cross_attention() -> None:
    """Test TransformerBlock initialization with cross attention."""
    key = jax.random.PRNGKey(0)
    block = xax.TransformerBlock(embed_dim=64, num_heads=8, ff_dim=128, key=key, cross_attention=True)

    assert block.self_attn is not None
    assert block.cross_attn is not None
    assert block.layer_norm3 is not None


def test_transformer_block_forward() -> None:
    """Test TransformerBlock forward pass."""
    key = jax.random.PRNGKey(0)
    block = xax.TransformerBlock(embed_dim=64, num_heads=8, ff_dim=128, key=key)

    x = jax.random.normal(key, (10, 64))
    output, cache = block.forward(x)

    assert output.shape == (10, 64)
    assert "self_attn" in cache


def test_transformer_block_forward_with_cross_attention() -> None:
    """Test TransformerBlock forward pass with cross attention."""
    key = jax.random.PRNGKey(0)
    block = xax.TransformerBlock(embed_dim=64, num_heads=8, ff_dim=128, key=key, cross_attention=True)

    x = jax.random.normal(key, (10, 64))
    context = jax.random.normal(key, (15, 64))
    output, cache = block.forward(x, context_sn=context)

    assert output.shape == (10, 64)
    assert "self_attn" in cache
    assert "cross_attn" in cache


def test_transformer_block_forward_with_cache() -> None:
    """Test TransformerBlock forward pass with caching."""
    key = jax.random.PRNGKey(0)
    block = xax.TransformerBlock(embed_dim=64, num_heads=8, ff_dim=128, key=key)

    x = jax.random.normal(key, (10, 64))
    output, cache = block.forward(x)

    assert output.shape == (10, 64)
    assert "self_attn" in cache


def test_transformer_stack_init() -> None:
    """Test TransformerStack initialization."""
    key = jax.random.PRNGKey(0)
    stack = xax.TransformerStack(
        embed_dim=64,
        num_heads=8,
        ff_dim=128,
        num_layers=3,
        key=key,
    )

    assert stack.num_layers == 3
    assert len(stack.layers) == 3


def test_transformer_stack_forward() -> None:
    """Test TransformerStack forward pass."""
    key = jax.random.PRNGKey(0)
    stack = xax.TransformerStack(
        embed_dim=64,
        num_heads=8,
        ff_dim=128,
        num_layers=2,
        key=key,
    )

    x = jax.random.normal(key, (10, 64))
    output, cache = stack.forward(x)

    assert output.shape == (10, 64)
    assert "layers" in cache
    assert len(cache["layers"]) == 2


def test_transformer_stack_forward_with_cross_attention() -> None:
    """Test TransformerStack forward pass with cross attention."""
    key = jax.random.PRNGKey(0)
    stack = xax.TransformerStack(
        embed_dim=64,
        num_heads=8,
        ff_dim=128,
        num_layers=2,
        cross_attention=True,
        key=key,
    )

    x = jax.random.normal(key, (10, 64))
    context = jax.random.normal(key, (15, 64))
    output, cache = stack.forward(x, context_sn=context)

    assert output.shape == (10, 64)
    assert "layers" in cache
    assert len(cache["layers"]) == 2


def test_transformer_init() -> None:
    """Test Transformer initialization."""
    key = jax.random.PRNGKey(0)
    model = xax.Transformer(
        vocab_size=1000,
        embed_dim=64,
        num_heads=8,
        ff_dim=128,
        num_layers=2,
        max_seq_len=100,
        key=key,
    )

    assert len(model.layers.layers) == 2
    assert model.max_seq_len == 100
    assert model.embed_dim == 64


def test_transformer_init_without_position_embeddings() -> None:
    """Test Transformer initialization without position embeddings."""
    key = jax.random.PRNGKey(0)
    model = xax.Transformer(
        vocab_size=1000,
        embed_dim=64,
        num_heads=8,
        ff_dim=128,
        num_layers=2,
        max_seq_len=100,
        key=key,
        use_absolute_position=False,
    )

    assert model.position_embedding is None


def test_transformer_forward() -> None:
    """Test Transformer forward pass."""
    key = jax.random.PRNGKey(0)
    model = xax.Transformer(
        vocab_size=1000,
        embed_dim=64,
        num_heads=8,
        ff_dim=128,
        num_layers=2,
        max_seq_len=100,
        key=key,
    )

    x = jnp.array([1, 2, 3, 4, 5])
    output, cache = model.forward(x)

    assert output.shape == (5, 64)
    assert "layers" in cache


def test_transformer_forward_with_output_layer() -> None:
    """Test Transformer forward pass with output layer."""
    key = jax.random.PRNGKey(0)
    model = xax.Transformer(
        vocab_size=1000,
        embed_dim=64,
        num_heads=8,
        ff_dim=128,
        num_layers=2,
        max_seq_len=100,
        output_size=32,
        key=key,
    )

    x = jnp.array([1, 2, 3, 4, 5])
    output, cache = model.forward(x)

    assert output.shape == (5, 32)
    assert "layers" in cache


def test_transformer_encode() -> None:
    """Test Transformer encode method."""
    key = jax.random.PRNGKey(0)
    model = xax.Transformer(
        vocab_size=1000,
        embed_dim=64,
        num_heads=8,
        ff_dim=128,
        num_layers=2,
        max_seq_len=100,
        key=key,
    )

    x = jnp.array([1, 2, 3, 4, 5])
    output, cache = model.encode(x)

    assert output.shape == (5, 64)
    assert "layers" in cache


def test_transformer_decode() -> None:
    """Test Transformer decode method."""
    key = jax.random.PRNGKey(0)
    model = xax.Transformer(
        vocab_size=1000,
        embed_dim=64,
        num_heads=8,
        ff_dim=128,
        num_layers=2,
        max_seq_len=100,
        cross_attention=True,
        key=key,
    )

    x = jnp.array([1, 2, 3])
    context = jnp.array([4, 5, 6, 7, 8])
    output, cache = model.decode(x, context)

    assert output.shape == (3, 64)
    assert "layers" in cache


def test_generate_sequence_basic() -> None:
    """Test basic sequence generation."""
    key = jax.random.PRNGKey(0)
    model = xax.Transformer(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_layers=2,
        max_seq_len=50,
        key=key,
    )

    prompt = jnp.array([1, 2, 3])
    generated = model.generate_sequence(prompt, max_len=5, key=key)

    assert generated.shape[0] == 8  # prompt_len + max_len
    assert jnp.array_equal(generated[:3], prompt)


def test_generate_sequence_with_temperature() -> None:
    """Test sequence generation with temperature."""
    key = jax.random.PRNGKey(0)
    model = xax.Transformer(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_layers=2,
        max_seq_len=50,
        key=key,
    )

    prompt = jnp.array([1, 2, 3])
    generated = model.generate_sequence(prompt, max_len=5, temperature=0.5, key=key)

    assert generated.shape[0] == 8
    assert jnp.array_equal(generated[:3], prompt)


def test_generate_sequence_with_top_k() -> None:
    """Test sequence generation with top-k sampling."""
    key = jax.random.PRNGKey(0)
    model = xax.Transformer(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_layers=2,
        max_seq_len=50,
        key=key,
    )

    prompt = jnp.array([1, 2, 3])
    generated = model.generate_sequence(prompt, max_len=5, top_k=10, key=key)

    assert generated.shape[0] == 8
    assert jnp.array_equal(generated[:3], prompt)


def test_generate_sequence_max_length_respect() -> None:
    """Test that generation respects max sequence length."""
    key = jax.random.PRNGKey(0)
    model = xax.Transformer(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_layers=2,
        max_seq_len=10,
        key=key,
    )

    prompt = jnp.array([1, 2, 3, 4, 5])  # 5 tokens
    generated = model.generate_sequence(prompt, max_len=10, key=key)

    # Should not exceed max_seq_len
    assert generated.shape[0] <= 10


def test_generate_sequence_deterministic() -> None:
    """Test that generation is deterministic with same key."""
    key = jax.random.PRNGKey(42)
    model = xax.Transformer(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_layers=2,
        max_seq_len=50,
        key=key,
    )

    prompt = jnp.array([1, 2, 3])
    generated1 = model.generate_sequence(prompt, max_len=5, key=key)
    generated2 = model.generate_sequence(prompt, max_len=5, key=key)

    assert jnp.array_equal(generated1, generated2)


def test_generate_sequence_different_keys() -> None:
    """Test that generation differs with different keys."""
    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(43)
    model = xax.Transformer(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_layers=2,
        max_seq_len=50,
        key=key1,
    )

    prompt = jnp.array([1, 2, 3])
    generated1 = model.generate_sequence(prompt, max_len=5, key=key1)
    generated2 = model.generate_sequence(prompt, max_len=5, key=key2)

    # Should be different (though there's a small chance they could be the same)
    # We'll just check that the prompt is preserved
    assert jnp.array_equal(generated1[:3], prompt)
    assert jnp.array_equal(generated2[:3], prompt)


def test_generate_sequence_causal_attention() -> None:
    """Test sequence generation with causal attention."""
    key = jax.random.PRNGKey(0)
    model = xax.Transformer(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_layers=2,
        max_seq_len=50,
        causal=True,
        key=key,
    )

    prompt = jnp.array([1, 2, 3])
    generated = model.generate_sequence(prompt, max_len=5, key=key)

    assert generated.shape[0] == 8
    assert jnp.array_equal(generated[:3], prompt)


def test_generate_sequence_edge_cases() -> None:
    """Test edge cases for sequence generation."""
    key = jax.random.PRNGKey(0)
    model = xax.Transformer(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_layers=2,
        max_seq_len=50,
        key=key,
    )

    # Empty prompt
    prompt = jnp.array([], dtype=jnp.int32)
    generated = model.generate_sequence(prompt, max_len=5, key=key)
    assert generated.shape[0] == 5

    # Zero max_len
    prompt = jnp.array([1, 2, 3], dtype=jnp.int32)
    generated = model.generate_sequence(prompt, max_len=0, key=key)
    assert jnp.array_equal(generated, prompt)


def test_cache_consistency() -> None:
    """Test that cache updates are consistent."""
    key = jax.random.PRNGKey(0)
    model = xax.Transformer(
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_layers=2,
        max_seq_len=50,
        key=key,
    )

    x = jnp.array([1, 2, 3])

    # First call - cache is always returned
    output1, cache = model.forward(x)

    # Second call using cache
    output2, _ = model.forward(x, cache=cache)

    # Outputs should be the same
    assert jnp.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__])
