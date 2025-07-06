"""Attention mechanisms for transformer models."""

from typing import NotRequired, TypedDict

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class Cache(TypedDict):
    k: Array
    v: Array


class CacheDict(TypedDict):
    self_attn: Cache
    cross_attn: NotRequired[Cache]


class TransformerCache(TypedDict):
    """Cache for the entire transformer stack."""

    layers: dict[str, CacheDict]


class SelfAttentionBlock(eqx.Module):
    """Self-attention block using jax.nn.dot_product_attention."""

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    num_heads: int = eqx.static_field()
    head_dim: int = eqx.static_field()
    causal: bool = eqx.static_field()
    context_length: int | None = eqx.static_field()

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        causal: bool = False,
        context_length: int | None = None,
    ) -> None:
        keys = jax.random.split(key, 4)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[0])
        self.k_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[1])
        self.v_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[2])
        self.output_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[3])

        self.causal = causal
        self.context_length = context_length

    def _reshape_for_multihead(self, x: Array) -> Array:
        """Reshape from (seq_len, embed_dim) to (seq_len, num_heads, head_dim)."""
        seq_len, _ = x.shape
        return x.reshape(seq_len, self.num_heads, self.head_dim)

    def _combine_heads(self, x: Array) -> Array:
        """Reshape from (seq_len, num_heads, head_dim) to (seq_len, embed_dim)."""
        _, n, h = x.shape
        return x.reshape(-1, n * h)

    def init_cache(self, dtype: jnp.dtype | None = None) -> Cache:
        """Initialize cache for the input.

        Args:
            dtype: The dtype of the cache

        Returns:
            Cache with fixed-length k and v tensors
        """
        if self.context_length is None:
            raise ValueError("context_length must be set for caching")

        # Create fixed-length cache
        k_cache = jnp.zeros((self.context_length, self.num_heads, self.head_dim), dtype=dtype)
        v_cache = jnp.zeros((self.context_length, self.num_heads, self.head_dim), dtype=dtype)

        return {"k": k_cache, "v": v_cache}

    def init_mask(self, seq_len: int) -> Array:
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        if self.context_length is not None:
            neg_mask = 1 - jnp.tril(jnp.ones((seq_len, seq_len)), -self.context_length)
            mask = mask * neg_mask
        return mask.astype(jnp.bool_)

    def forward(
        self,
        x_tn: Array,
        *,
        mask: Array | None = None,
        cache: Cache | None = None,
    ) -> tuple[Array, Cache]:
        """Apply self-attention.

        Args:
            x_tn: Input tensor of shape (seq_len, embed_dim)
            mask: Optional mask tensor
            cache: The cached key and value tensors (fixed-length)

        Returns:
            The output tensor of shape (seq_len, embed_dim) and updated cache
        """
        chex.assert_rank(x_tn, 2)

        # Project inputs to queries, keys, and values
        q = jax.vmap(self.q_proj)(x_tn)
        k = jax.vmap(self.k_proj)(x_tn)
        v = jax.vmap(self.v_proj)(x_tn)

        # Reshape to multihead format
        q = self._reshape_for_multihead(q)
        k = self._reshape_for_multihead(k)
        v = self._reshape_for_multihead(v)

        if cache is not None:
            k_cache = cache["k"]
            v_cache = cache["v"]
            seq_len = k.shape[0]

            # Roll left by seq_len, insert new k/v at the end
            k_cache = jnp.roll(k_cache, -seq_len, axis=0).at[-seq_len:].set(k)
            v_cache = jnp.roll(v_cache, -seq_len, axis=0).at[-seq_len:].set(v)

            # Use the updated cache for attention
            k_attn = k_cache
            v_attn = v_cache

        else:
            k_cache = k
            v_cache = v
            k_attn = k
            v_attn = v

        attn_output = jax.nn.dot_product_attention(
            q,
            k_attn,
            v_attn,
            mask=mask,
            is_causal=self.causal and mask is None,
        )

        attn_output = self._combine_heads(attn_output)
        output = jax.vmap(self.output_proj)(attn_output)

        return output, {"k": k_cache, "v": v_cache}


class CrossAttentionBlock(eqx.Module):
    """Cross-attention block using jax.nn.dot_product_attention."""

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    num_heads: int = eqx.static_field()
    head_dim: int = eqx.static_field()

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        keys = jax.random.split(key, 4)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[0])
        self.k_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[1])
        self.v_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[2])
        self.output_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[3])

    def _reshape_for_multihead(self, x: Array) -> Array:
        """Reshape from (seq_len, embed_dim) to (seq_len, num_heads, head_dim)."""
        seq_len, _ = x.shape
        return x.reshape(seq_len, self.num_heads, self.head_dim)

    def _combine_heads(self, x: Array) -> Array:
        """Reshape from (seq_len, num_heads, head_dim) to (seq_len, embed_dim)."""
        seq_len, _, _ = x.shape
        return x.reshape(seq_len, -1)

    def init_cache(self, x_tn: Array) -> Cache:
        """Initialize cache for the input."""
        chex.assert_rank(x_tn, 2)
        k = jax.vmap(self.k_proj)(x_tn)
        v = jax.vmap(self.v_proj)(x_tn)
        k = self._reshape_for_multihead(k)
        v = self._reshape_for_multihead(v)
        return {"k": k, "v": v}

    def forward(
        self,
        q_tn: Array,
        *,
        kv_sn: Array | None = None,
        cache: Cache | None = None,
        mask: Array | None = None,
    ) -> tuple[Array, Cache]:
        """Apply cross-attention.

        Args:
            q_tn: Query input tensor of shape (q_seq_len, embed_dim)
            kv_sn: Key/value input tensor of shape (kv_seq_len, embed_dim).
                If not provided, then `cache` must be provided.
            cache: The cached key and value tensors. If not provided, then
                `kv_input_sn` must be provided.
            mask: Optional mask tensor

        Returns:
            The output tensor of shape (q_seq_len, embed_dim)
        """
        chex.assert_rank(q_tn, 2)

        # Project inputs to queries, keys, and values
        q = jax.vmap(self.q_proj)(q_tn)

        # Use cached key/value if provided
        if cache is not None:
            k = cache["k"]
            v = cache["v"]
        elif kv_sn is not None:
            chex.assert_rank(kv_sn, 2)
            k = jax.vmap(self.k_proj)(kv_sn)
            v = jax.vmap(self.v_proj)(kv_sn)
        else:
            raise ValueError("Either `cache` or `kv_input_sn` must be provided.")

        # Reshape to multihead format
        q = self._reshape_for_multihead(q)
        k = self._reshape_for_multihead(k)
        v = self._reshape_for_multihead(v)

        # Apply dot product attention
        attn_output = jax.nn.dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            is_causal=False,
        )

        # Combine heads
        attn_output = self._combine_heads(attn_output)

        # Final projection
        output = jax.vmap(self.output_proj)(attn_output)

        return output, {"k": k, "v": v}


class TransformerBlock(eqx.Module):
    self_attn: SelfAttentionBlock
    cross_attn: CrossAttentionBlock | None
    feed_forward: eqx.nn.MLP
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    layer_norm3: eqx.nn.LayerNorm | None

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        key: PRNGKeyArray,
        causal: bool = False,
        cross_attention: bool = False,
        context_length: int | None = None,
    ) -> None:
        keys = jax.random.split(key, 3)

        self.self_attn = SelfAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            key=keys[0],
            causal=causal,
            context_length=context_length,
        )

        if cross_attention:
            self.cross_attn = CrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                key=keys[1],
            )
            self.layer_norm3 = eqx.nn.LayerNorm(embed_dim)

        else:
            self.cross_attn = None
            self.layer_norm3 = None

        self.layer_norm1 = eqx.nn.LayerNorm(embed_dim)
        self.layer_norm2 = eqx.nn.LayerNorm(embed_dim)

        self.feed_forward = eqx.nn.MLP(
            in_size=embed_dim,
            out_size=embed_dim,
            width_size=ff_dim,
            depth=1,
            activation=jax.nn.gelu,
            key=keys[2],
        )

    def init_cache(self, dtype: jnp.dtype | None = None, x_tn: Array | None = None) -> CacheDict:
        """Initialize cache for the input."""
        cache = {}
        if dtype is None and x_tn is not None:
            dtype = x_tn.dtype
        cache["self_attn"] = self.self_attn.init_cache(dtype=dtype)
        if self.cross_attn is not None:
            if x_tn is None:
                raise ValueError("x_tn must be provided if cross_attn is not None")
            cache["cross_attn"] = self.cross_attn.init_cache(x_tn=x_tn)
        return cache

    def init_mask(self, seq_len: int) -> Array:
        return self.self_attn.init_mask(seq_len)

    def forward(
        self,
        x_tn: Array,
        *,
        context_sn: Array | None = None,
        self_mask: Array | None = None,
        cross_mask: Array | None = None,
        cache: CacheDict | None = None,
    ) -> tuple[Array, CacheDict]:
        """Apply transformer block.

        Args:
            x_tn: Input tensor of shape (seq_len, embed_dim)
            context_sn: Optional context for cross-attention
            self_mask: Mask for self-attention
            cross_mask: Mask for cross-attention
            cache: Optional dictionary containing cached key and value tensors

        Returns:
            The output tensor and the updated cache
        """
        chex.assert_rank(x_tn, 2)

        # Initialize cache if needed
        updated_cache: CacheDict = {}
        if cache is None:
            cache = {}

        # Self-attention block with pre-norm
        norm_x = jax.vmap(self.layer_norm1)(x_tn)

        attn_output, updated_cache["self_attn"] = self.self_attn.forward(
            x_tn=norm_x,
            mask=self_mask,
            cache=cache.get("self_attn"),
        )

        x_tn = x_tn + attn_output

        # Cross-attention block (if enabled) with pre-norm
        if self.cross_attn is not None and context_sn is not None:
            assert self.layer_norm3 is not None

            norm_x = jax.vmap(self.layer_norm3)(x_tn)

            cross_attn_output, updated_cache["cross_attn"] = self.cross_attn.forward(
                q_tn=norm_x,
                kv_sn=context_sn,
                mask=cross_mask,
                cache=cache.get("cross_attn"),
            )

            x_tn = x_tn + cross_attn_output

        elif context_sn is not None:
            raise ValueError("Cross-attention is enabled but context is not provided.")

        elif self.cross_attn is not None:
            raise ValueError("Cross-attention is enabled but context is not provided.")

        # Feed-forward block with pre-norm
        norm_x = jax.vmap(self.layer_norm2)(x_tn)
        ff_output = jax.vmap(self.feed_forward)(norm_x)
        x_tn = x_tn + ff_output

        return x_tn, updated_cache


class TransformerStack(eqx.Module):
    """A stack of transformer blocks."""

    layers: list[TransformerBlock]
    num_layers: int = eqx.static_field()
    causal: bool = eqx.static_field()

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        *,
        key: PRNGKeyArray,
        causal: bool = False,
        cross_attention: bool = False,
        context_length: int | None = None,
    ) -> None:
        keys = jax.random.split(key, num_layers)

        self.layers = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                key=keys[i],
                causal=causal,
                cross_attention=cross_attention,
                context_length=context_length,
            )
            for i in range(num_layers)
        ]

        self.num_layers = num_layers
        self.causal = causal

    def init_cache(self, dtype: jnp.dtype | None = None, x_tn: Array | None = None) -> TransformerCache:
        """Initialize cache for the input."""
        cache = {}
        for i, layer in enumerate(self.layers):
            cache[f"layer_{i}"] = layer.init_cache(dtype=dtype, x_tn=x_tn)
        return {"layers": cache}

    def init_mask(self, seq_len: int) -> Array:
        return self.layers[0].init_mask(seq_len)

    def forward(
        self,
        x_tn: Array,
        *,
        context_sn: Array | None = None,
        self_mask: Array | None = None,
        cross_mask: Array | None = None,
        cache: TransformerCache | None = None,
    ) -> tuple[Array, TransformerCache]:
        """Apply transformer stack.

        Args:
            x_tn: Input tensor of shape (seq_len, embed_dim)
            context_sn: Optional context for cross-attention
            self_mask: Mask for self-attention
            cross_mask: Mask for cross-attention
            cache: Optional dictionary containing cached key and value tensors

        Returns:
            The output tensor and the updated cache
        """
        # Initialize layer caches
        if cache is None:
            cache = {"layers": {f"layer_{i}": {} for i in range(self.num_layers)}}

        # Updated cache will be built
        updated_cache = {"layers": {}}

        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache["layers"][f"layer_{i}"]

            x_tn, layer_updated_cache = layer.forward(
                x_tn,
                context_sn=context_sn,
                self_mask=self_mask,
                cross_mask=cross_mask,
                cache=layer_cache,
            )
            updated_cache["layers"][f"layer_{i}"] = layer_updated_cache

        return x_tn, updated_cache


class Transformer(eqx.Module):
    token_embedding: eqx.nn.Embedding
    position_embedding: eqx.nn.Embedding | None
    layers: TransformerStack
    output_layer: eqx.nn.Linear | None
    layer_norm: eqx.nn.LayerNorm
    max_seq_len: int = eqx.static_field()
    embed_dim: int = eqx.static_field()
    causal: bool = eqx.static_field()

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        max_seq_len: int,
        output_size: int | None = None,
        *,
        key: PRNGKeyArray,
        causal: bool = False,
        cross_attention: bool = False,
        context_length: int | None = None,
        use_absolute_position: bool = True,
    ) -> None:
        # Calculate number of keys needed
        num_keys = 3 if output_size is None else 4
        keys = jax.random.split(key, num_keys)

        self.token_embedding = eqx.nn.Embedding(vocab_size, embed_dim, key=keys[0])

        # Position embeddings can be disabled
        if use_absolute_position:
            self.position_embedding = eqx.nn.Embedding(max_seq_len, embed_dim, key=keys[1])
        else:
            self.position_embedding = None

        self.layers = TransformerStack(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            key=keys[2],
            causal=causal,
            cross_attention=cross_attention,
            context_length=context_length,
        )

        self.layer_norm = eqx.nn.LayerNorm(embed_dim)
        if output_size is not None:
            self.output_layer = eqx.nn.Linear(embed_dim, output_size, key=keys[3])
        else:
            self.output_layer = None

        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.causal = causal

    def _add_positional_embedding(self, x_embedded: Array, positions: Array | None = None) -> Array:
        """Add positional embeddings to the token embeddings."""
        if self.position_embedding is None:
            return x_embedded

        seq_len, _ = x_embedded.shape

        if positions is None:
            positions = jnp.arange(seq_len)
        pos_embedded = jax.vmap(self.position_embedding)(positions)

        return x_embedded + pos_embedded

    def init_cache(self, dtype: jnp.dtype | None = None, x_tn: Array | None = None) -> TransformerCache:
        """Initialize cache for the input."""
        return self.layers.init_cache(dtype=dtype, x_tn=x_tn)

    def init_mask(self, seq_len: int) -> Array:
        return self.layers.init_mask(seq_len)

    def encode(
        self,
        x: Array,
        *,
        mask: Array | None = None,
        positions: Array | None = None,
        cache: TransformerCache | None = None,
    ) -> tuple[Array, TransformerCache]:
        """Encode the input sequence.

        Args:
            x: Input token indices of shape (seq_len)
            mask: Optional attention mask
            positions: Optional positions
            cache: Optional dictionary containing cached key and value tensors

        Returns:
            The encoded representation and the updated cache
        """
        # Token embedding
        x_embedded = jax.vmap(self.token_embedding)(x)

        # Add positional embedding
        x_embedded = self._add_positional_embedding(x_embedded, positions)

        # Apply transformer stack
        x_embedded, updated_cache = self.layers.forward(
            x_embedded,
            self_mask=mask,
            cache=cache,
        )

        # Apply final layer norm
        output = jax.vmap(self.layer_norm)(x_embedded)

        return output, updated_cache

    def decode(
        self,
        x_t: Array,
        context_s: Array,
        *,
        self_mask: Array | None = None,
        cross_mask: Array | None = None,
        positions: Array | None = None,
        cache: TransformerCache | None = None,
    ) -> tuple[Array, TransformerCache]:
        """Decode with self-attention and cross-attention.

        Args:
            x_t: Input token indices, shape (seq_len)
            context_s: Context from encoder (token indices or embedded),
                shape (context_len, embed_dim)
            self_mask: Optional self-attention mask, shape (seq_len, seq_len)
            cross_mask: Optional cross-attention mask, shape (seq_len, context_len)
            positions: Optional positions, shape (seq_len)
            cache: Optional dictionary containing cached key and value tensors

        Returns:
            The decoded representation and the updated cache
        """
        # Token embedding for x
        x_embedded = jax.vmap(self.token_embedding)(x_t)
        x_embedded = self._add_positional_embedding(x_embedded, positions)

        # Token embedding for context if needed
        context_embedded = jax.vmap(self.token_embedding)(context_s)
        context_embedded = self._add_positional_embedding(context_embedded)

        # Apply transformer stack with cross-attention
        x_embedded, updated_cache = self.layers.forward(
            x_embedded,
            context_sn=context_embedded,
            self_mask=self_mask,
            cross_mask=cross_mask,
            cache=cache,
        )

        # Apply final layer norm
        output = jax.vmap(self.layer_norm)(x_embedded)

        return output, updated_cache

    def forward(
        self,
        x: Array,
        *,
        mask: Array | None = None,
        positions: Array | None = None,
        cache: TransformerCache | None = None,
    ) -> tuple[Array, TransformerCache]:
        """Forward pass for encoder-only or decoder-only transformers.

        Args:
            x: Input token indices of shape (seq_len)
            mask: Optional attention mask
            positions: Optional positions
            cache: Optional dictionary containing cached key and value tensors

        Returns:
            The output representation and the updated cache
        """
        chex.assert_rank(x, 1)

        output, updated_cache = self.encode(
            x,
            mask=mask,
            positions=positions,
            cache=cache,
        )

        # Apply output layer if it exists
        if self.output_layer is not None:
            output = jax.vmap(self.output_layer)(output)

        return output, updated_cache

    def predict_sequence(self, x_seq: Array) -> Array:
        output, _ = self.forward(x=x_seq)
        return output

    def generate_sequence(
        self,
        prompt_seq: Array,
        max_len: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Array:
        """Generate a sequence autoregressively with KV caching.

        Args:
            prompt_seq: Input token indices of shape (prompt_len,)
            max_len: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Optional top-k sampling parameter
            key: PRNG key for sampling

        Returns:
            Generated sequence of shape (prompt_len + max_len,)
        """
        if not self.causal:
            raise ValueError("generate_sequence is only supported for causal models")

        if key is None:
            key = jax.random.PRNGKey(0)

        prompt_len = prompt_seq.shape[0]
        max_gen_len = min(max_len, self.max_seq_len - prompt_len)
        if max_gen_len <= 0:
            return prompt_seq

        total_len = prompt_len + max_gen_len
        output_seq = jnp.zeros(total_len, dtype=prompt_seq.dtype)
        output_seq = output_seq.at[:prompt_len].set(prompt_seq)

        # Initialize cache with prompt
        _, cache = self.encode(prompt_seq)

        # Define scan function for autoregressive generation
        def scan_fn(
            carry: tuple[Array, int, TransformerCache, PRNGKeyArray],
            _: Array,
        ) -> tuple[tuple[Array, int, TransformerCache, PRNGKeyArray], Array]:
            output_seq, pos, cache, rng = carry
            current_token = jax.lax.dynamic_slice(output_seq, (pos,), (1,))
            pos_tensor = jnp.array([pos])

            # Forward pass with cache update
            logits, new_cache = self.forward(
                x=current_token,
                positions=pos_tensor,
                cache=cache,
            )

            logits = logits[-1] / temperature
            if top_k is not None:
                top_logits, top_indices = jax.lax.top_k(logits, top_k)
                logits = jnp.full_like(logits, float("-inf"))
                logits = logits.at[top_indices].set(top_logits)
            rng, subrng = jax.random.split(rng)
            next_token = jax.random.categorical(subrng, logits[None, ...])[0]
            new_output_seq = jax.lax.dynamic_update_slice(output_seq, next_token[None], (pos + 1,))

            return (new_output_seq, pos + 1, new_cache, rng), next_token

        init_carry = (output_seq, prompt_len - 1, cache, key)
        (final_seq, _, _, _), _ = jax.lax.scan(scan_fn, init_carry, jnp.arange(max_gen_len))
        return final_seq
