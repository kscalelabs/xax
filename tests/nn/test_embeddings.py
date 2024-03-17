"""Tests the embeddings API."""

from typing import get_args

import jax.numpy as jnp
import jax.random as jrandom
import pytest

import xax


@pytest.mark.parametrize("kind", get_args(xax.EmbeddingKind))
def test_embeddings_api(kind: xax.EmbeddingKind) -> None:
    key = jrandom.PRNGKey(0)
    xkey, pkey = jrandom.split(key)
    x = jrandom.normal(xkey, (5, 8), dtype=jnp.double)
    times_t = jnp.arange(1, 6)
    emb = xax.get_positional_embeddings(kind, max_tsz=12, embed_dim=8, key=pkey)
    y1 = emb(x, times_t=times_t)
    y2 = emb(x, offset=1)
    assert y1.shape == (5, 8)
    assert y2.shape == (5, 8)
    assert jnp.allclose(y1, y2)


@pytest.mark.parametrize("offset", [0, 12])
def test_rotary_embeddings_inference(offset: int) -> None:
    key = jrandom.PRNGKey(0)
    x = jrandom.normal(key, (5, 8), dtype=jnp.double)
    emb = xax.get_positional_embeddings("rotary", max_tsz=8 + offset, embed_dim=8, learnable=False)
    y1 = emb(x, offset=offset)
    y2 = xax.rotary_embeddings(x, offset=offset)
    assert y1.shape == (5, 8)
    assert y2.shape == (5, 8)
    assert jnp.allclose(y1, y2)
