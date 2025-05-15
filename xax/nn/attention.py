"""Attention mechanisms for transformer models."""

from typing import Literal, cast, overload

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class SelfAttentionBlock(eqx.Module):
    """Self-attention block using jax.nn.dot_product_attention."""

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    num_heads: int = eqx.static_field()
    head_dim: int = eqx.static_field()
    causal: bool = eqx.static_field()

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        causal: bool = False,
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

    def _reshape_for_multihead(self, x: Array) -> Array:
        """Reshape from (seq_len, embed_dim) to (seq_len, num_heads, head_dim)."""
        seq_len, _ = x.shape
        return x.reshape(seq_len, self.num_heads, self.head_dim)

    def _combine_heads(self, x: Array) -> Array:
        """Reshape from (seq_len, num_heads, head_dim) to (seq_len, embed_dim)."""
        seq_len, _, _ = x.shape
        return x.reshape(seq_len, -1)

    def __call__(
        self,
        x: Array,
        *,
        key: PRNGKeyArray | None = None,
        mask: Array | None = None,
        cache: dict[str, Array] | None = None,
        update_cache: bool = False,
    ) -> Array | tuple[Array, dict[str, Array]]:
        """Apply self-attention to the input.

        Args:
            x: Input tensor of shape (seq_len, embed_dim)
            key: PRNGKey for dropout randomness
            mask: Optional mask tensor of shape (seq_len, seq_len) or broadcastable
            cache: Optional dictionary containing cached key and value tensors
            update_cache: Whether to update the cache and return it

        Returns:
            If update_cache is False: Output tensor of shape (seq_len, embed_dim)
            If update_cache is True: Tuple of (output tensor, updated cache)
        """
        chex.assert_rank(x, 2)

        # Project inputs to queries, keys, and values
        q = jax.vmap(self.q_proj)(x)

        # Use cached key/value if provided and not updating cache
        if cache is not None and not update_cache:
            k = cache["k"]
            v = cache["v"]
        else:
            k = jax.vmap(self.k_proj)(x)
            v = jax.vmap(self.v_proj)(x)

            # Update cache if needed
            if update_cache:
                if cache is None:
                    cache = {}
                cache = {"k": k, "v": v}

        # Reshape to multihead format
        q = self._reshape_for_multihead(q)
        k = self._reshape_for_multihead(k)
        v = self._reshape_for_multihead(v)

        # Apply dot product attention.
        # Note that Apple Silicon struggles with this:
        # https://github.com/jax-ml/jax/issues/20114
        attn_output = jax.nn.dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            is_causal=self.causal and mask is None,
        )

        # Combine heads
        attn_output = self._combine_heads(attn_output)

        # Final projection
        output = jax.vmap(self.output_proj)(attn_output)

        if update_cache:
            return output, cast(dict[str, Array], cache)
        return output


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

    def __call__(
        self,
        q_input: Array,
        kv_input: Array,
        *,
        key: PRNGKeyArray | None = None,
        mask: Array | None = None,
        cache: dict[str, Array] | None = None,
        update_cache: bool = False,
    ) -> Array | tuple[Array, dict[str, Array]]:
        """Apply cross-attention.

        Args:
            q_input: Query input tensor of shape (q_seq_len, embed_dim)
            kv_input: Key/value input tensor of shape (kv_seq_len, embed_dim)
            key: PRNGKey for dropout randomness
            mask: Optional mask tensor
            cache: Optional dictionary containing cached key and value tensors
            update_cache: Whether to update the cache and return it

        Returns:
            If update_cache is False: Output tensor of shape (q_seq_len, embed_dim)
            If update_cache is True: Tuple of (output tensor, updated cache)
        """
        chex.assert_rank(q_input, 2)
        chex.assert_rank(kv_input, 2)

        # Project inputs to queries, keys, and values
        q = jax.vmap(self.q_proj)(q_input)

        # Use cached key/value if provided and not updating cache
        if cache is not None and not update_cache:
            k = cache["k"]
            v = cache["v"]
        else:
            k = jax.vmap(self.k_proj)(kv_input)
            v = jax.vmap(self.v_proj)(kv_input)

            # Update cache if needed
            if update_cache:
                if cache is None:
                    cache = {}
                cache = {"k": k, "v": v}

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

        if update_cache:
            return output, cast(dict[str, Array], cache)
        return output


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
    ) -> None:
        keys = jax.random.split(key, 4)

        self.self_attn = SelfAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            key=keys[0],
            causal=causal,
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

    @overload
    def __call__(
        self,
        x: Array,
        *,
        context: Array | None = None,
        self_mask: Array | None = None,
        cross_mask: Array | None = None,
        key: PRNGKeyArray | None = None,
        cache: dict[str, dict[str, Array]] | None = None,
        update_cache: Literal[True],
    ) -> tuple[Array, dict[str, dict[str, Array]]]: ...

    @overload
    def __call__(
        self,
        x: Array,
        *,
        context: Array | None = None,
        self_mask: Array | None = None,
        cross_mask: Array | None = None,
        key: PRNGKeyArray | None = None,
        cache: dict[str, dict[str, Array]] | None = None,
        update_cache: Literal[False] = False,
    ) -> Array: ...

    def __call__(
        self,
        x: Array,
        *,
        context: Array | None = None,
        self_mask: Array | None = None,
        cross_mask: Array | None = None,
        key: PRNGKeyArray | None = None,
        cache: dict[str, dict[str, Array]] | None = None,
        update_cache: bool = False,
    ) -> Array | tuple[Array, dict[str, dict[str, Array]]]:
        """Apply transformer block.

        Args:
            x: Input tensor
            context: Optional context for cross-attention
            self_mask: Mask for self-attention
            cross_mask: Mask for cross-attention
            key: Optional PRNG key for dropout
            cache: Optional dictionary containing cached key and value tensors
            update_cache: Whether to update the cache and return it

        Returns:
            If update_cache is False: Output tensor
            If update_cache is True: Tuple of (output tensor, updated cache)
        """
        chex.assert_rank(x, 2)
        if key is not None:
            key1, key2 = jax.random.split(key)
        else:
            key1 = key2 = None

        # Initialize cache if needed
        updated_cache = {}
        if cache is None:
            cache = {}

        # Self-attention block with pre-norm
        norm_x = jax.vmap(self.layer_norm1)(x)

        self_attn_cache = cache.get("self_attn")
        if update_cache:
            attn_output, self_attn_cache = self.self_attn(
                norm_x, key=key1, mask=self_mask, cache=self_attn_cache, update_cache=True
            )
            updated_cache["self_attn"] = self_attn_cache
        else:
            attn_output = self.self_attn(norm_x, key=key1, mask=self_mask, cache=self_attn_cache)

        x = x + attn_output

        # Cross-attention block (if enabled) with pre-norm
        if self.cross_attn is not None and context is not None:
            assert self.layer_norm3 is not None

            norm_x = jax.vmap(self.layer_norm3)(x)
            cross_attn_cache = cache.get("cross_attn")

            if update_cache:
                cross_attn_output, cross_attn_cache = self.cross_attn(
                    norm_x, context, key=key2, mask=cross_mask, cache=cross_attn_cache, update_cache=True
                )
                updated_cache["cross_attn"] = cross_attn_cache
            else:
                cross_attn_output = self.cross_attn(norm_x, context, key=key2, mask=cross_mask, cache=cross_attn_cache)

            x = x + cross_attn_output

        # Feed-forward block with pre-norm
        norm_x = jax.vmap(self.layer_norm2)(x)
        ff_output = jax.vmap(self.feed_forward)(norm_x)
        x = x + ff_output

        if update_cache:
            return x, updated_cache
        return x


class Transformer(eqx.Module):
    token_embedding: eqx.nn.Embedding
    position_embedding: eqx.nn.Embedding | None
    layers: list[TransformerBlock]
    output_layer: eqx.nn.Linear | None
    layer_norm: eqx.nn.LayerNorm
    max_seq_len: int = eqx.static_field()
    embed_dim: int = eqx.static_field()

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
        use_absolute_position: bool = True,
    ) -> None:
        keys = jax.random.split(key, num_layers + 3)

        self.token_embedding = eqx.nn.Embedding(vocab_size, embed_dim, key=keys[0])

        # Position embeddings can be disabled
        if use_absolute_position:
            self.position_embedding = eqx.nn.Embedding(max_seq_len, embed_dim, key=keys[1])
        else:
            self.position_embedding = None

        self.layers = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                key=keys[i + 2],
                causal=causal,
                cross_attention=cross_attention,
            )
            for i in range(num_layers)
        ]

        self.layer_norm = eqx.nn.LayerNorm(embed_dim)

        if output_size is not None:
            self.output_layer = eqx.nn.Linear(embed_dim, output_size, key=keys[-1])
        else:
            self.output_layer = None

        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

    def _add_positional_embedding(self, x_embedded: Array, positions: Array | None = None) -> Array:
        """Add positional embeddings to the token embeddings."""
        if self.position_embedding is None:
            return x_embedded

        seq_len, _ = x_embedded.shape

        if positions is None:
            positions = jnp.arange(seq_len)
        pos_embedded = jax.vmap(self.position_embedding)(positions)

        return x_embedded + pos_embedded

    def encode(
        self,
        x: Array,
        mask: Array | None = None,
        positions: Array | None = None,
        key: PRNGKeyArray | None = None,
        cache: dict[str, dict[str, dict[str, Array]]] | None = None,
        update_cache: bool = False,
    ) -> Array | tuple[Array, dict[str, dict[str, dict[str, Array]]]]:
        """Encode the input sequence.

        Args:
            x: Input token indices of shape (seq_len)
            mask: Optional attention mask
            positions: Optional positions
            key: Optional PRNG key for dropout
            cache: Optional dictionary containing cached key and value tensors
            update_cache: Whether to update the cache and return it

        Returns:
            If update_cache is False: Encoded representation
            If update_cache is True: Tuple of (encoded representation, updated cache)
        """
        # Token embedding
        x_embedded = jax.vmap(self.token_embedding)(x)

        # Add positional embedding
        x_embedded = self._add_positional_embedding(x_embedded, positions)

        # Initialize layer caches
        if cache is None and update_cache:
            cache = {f"layer_{i}": {} for i in range(len(self.layers))}

        # Updated cache will be built if needed
        updated_cache = {}

        # Apply transformer layers
        keys: Array | list[None] = [None] * len(self.layers)
        if key is not None:
            keys = jax.random.split(key, len(self.layers))

        for i, (layer, layer_key) in enumerate(zip(self.layers, keys, strict=False)):
            layer_cache = None if cache is None else cache.get(f"layer_{i}")

            if update_cache:
                x_embedded, layer_updated_cache = layer.__call__(
                    x_embedded,
                    self_mask=mask,
                    key=layer_key,
                    cache=layer_cache,
                    update_cache=True,
                )
                updated_cache[f"layer_{i}"] = layer_updated_cache
            else:
                x_embedded = layer.__call__(
                    x_embedded,
                    self_mask=mask,
                    key=layer_key,
                    cache=layer_cache,
                )

        # Apply final layer norm
        output = jax.vmap(self.layer_norm)(x_embedded)

        if update_cache:
            return output, updated_cache
        return output

    def decode(
        self,
        x: Array,
        context: Array,
        self_mask: Array | None = None,
        cross_mask: Array | None = None,
        positions: Array | None = None,
        key: PRNGKeyArray | None = None,
        cache: dict[str, dict[str, dict[str, Array]]] | None = None,
        update_cache: bool = False,
    ) -> Array | tuple[Array, dict[str, dict[str, dict[str, Array]]]]:
        """Decode with self-attention and cross-attention.

        Args:
            x: Input token indices
            context: Context from encoder
            self_mask: Optional self-attention mask
            cross_mask: Optional cross-attention mask
            positions: Optional positions
            key: Optional PRNG key for dropout
            cache: Optional dictionary containing cached key and value tensors
            update_cache: Whether to update the cache and return it

        Returns:
            If update_cache is False: Decoded representation
            If update_cache is True: Tuple of (decoded representation, updated cache)
        """
        # Token embedding
        x_embedded = jax.vmap(lambda x_seq: jax.vmap(self.token_embedding)(x_seq))(x)

        # Add positional embedding
        x_embedded = self._add_positional_embedding(x_embedded, positions)

        # Initialize layer caches
        if cache is None and update_cache:
            cache = {f"layer_{i}": {} for i in range(len(self.layers))}

        # Updated cache will be built if needed
        updated_cache = {}

        # Apply transformer layers with cross-attention
        keys: Array | list[None] = [None] * len(self.layers)
        if key is not None:
            keys = jax.random.split(key, len(self.layers))

        for i, (layer, layer_key) in enumerate(zip(self.layers, keys, strict=False)):
            layer_cache = None if cache is None else cache.get(f"layer_{i}")

            if update_cache:
                x_embedded, layer_updated_cache = layer.__call__(
                    x_embedded,
                    context=context,
                    self_mask=self_mask,
                    cross_mask=cross_mask,
                    key=layer_key,
                    cache=layer_cache,
                    update_cache=True,
                )
                updated_cache[f"layer_{i}"] = layer_updated_cache
            else:
                x_embedded = layer(
                    x_embedded,
                    context=context,
                    self_mask=self_mask,
                    cross_mask=cross_mask,
                    key=layer_key,
                    cache=layer_cache,
                )

        # Apply final layer norm
        output = jax.vmap(self.layer_norm)(x_embedded)

        if update_cache:
            return output, updated_cache
        return output

    @overload
    def __call__(
        self,
        x: Array,
        *,
        mask: Array | None = None,
        positions: Array | None = None,
        key: PRNGKeyArray | None = None,
        cache: dict[str, dict[str, dict[str, Array]]] | None = None,
        update_cache: Literal[True],
    ) -> tuple[Array, dict[str, dict[str, dict[str, Array]]]]: ...

    @overload
    def __call__(
        self,
        x: Array,
        *,
        mask: Array | None = None,
        positions: Array | None = None,
        key: PRNGKeyArray | None = None,
        cache: dict[str, dict[str, dict[str, Array]]] | None = None,
        update_cache: Literal[False] = False,
    ) -> Array: ...

    def __call__(
        self,
        x: Array,
        *,
        mask: Array | None = None,
        positions: Array | None = None,
        key: PRNGKeyArray | None = None,
        cache: dict[str, dict[str, dict[str, Array]]] | None = None,
        update_cache: bool = False,
    ) -> Array | tuple[Array, dict[str, dict[str, dict[str, Array]]]]:
        """Forward pass for encoder-only or decoder-only transformers.

        Args:
            x: Input token indices of shape (seq_len)
            mask: Optional attention mask
            positions: Optional positions
            key: Optional PRNG key for dropout
            cache: Optional dictionary containing cached key and value tensors
            update_cache: Whether to update the cache and return it

        Returns:
            If update_cache is False: Output representation
            If update_cache is True: Tuple of (output representation, updated cache)
        """
        chex.assert_rank(x, 1)

        if update_cache:
            output, updated_cache = self.encode(
                x, mask=mask, positions=positions, key=key, cache=cache, update_cache=True
            )
        else:
            output = self.encode(x, mask=mask, positions=positions, key=key, cache=cache)

        # Apply output layer if it exists
        if self.output_layer is not None:
            output = jax.vmap(self.output_layer)(output)

        if update_cache:
            return output, updated_cache
        return output

    def predict_sequence(self, x_seq: Array) -> Array:
        return self(x=x_seq)

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
        if key is None:
            key = jax.random.PRNGKey(0)

        prompt_len = prompt_seq.shape[0]
        sequence = prompt_seq

        # Create causal mask for generation
        causal_mask = jnp.tril(jnp.ones((self.max_seq_len, self.max_seq_len), dtype=jnp.bool_))

        # Initialize cache with the prompt
        _, cache = self(x=prompt_seq, mask=causal_mask[:prompt_len, :prompt_len], update_cache=True)

        # Define decode step function (for clarity)
        def decode_step(seq: Array, pos: int, cur_cache: dict, rng: PRNGKeyArray) -> tuple[Array, dict, PRNGKeyArray]:
            # Get the next position and last token
            pos_tensor = jnp.array([pos])
            last_token = seq[-1:]

            # Get logits for next token
            rng, subrng = jax.random.split(rng)
            logits, new_cache = self(
                x=last_token,
                positions=pos_tensor,
                key=subrng,
                cache=cur_cache,
                update_cache=True,
            )

            # Extract final logits and apply temperature
            logits = logits[-1] / temperature

            # Apply top-k sampling if specified
            if top_k is not None:
                top_logits, top_indices = jax.lax.top_k(logits, top_k)
                logits = jnp.full_like(logits, float("-inf"))
                logits = logits.at[top_indices].set(top_logits)

            # Sample next token
            rng, subrng = jax.random.split(rng)
            next_token = jax.random.categorical(subrng, logits[None, ...])[0]

            # Add token to sequence
            new_seq = jnp.concatenate([seq, next_token[None]], axis=0)
            return new_seq, new_cache, rng

        # Generate tokens one by one
        for _ in range(max_len):
            # Break if max sequence length reached
            if sequence.shape[0] >= self.max_seq_len:
                break

            # Decode next token
            sequence, cache, key = decode_step(seq=sequence, pos=sequence.shape[0] - 1, cur_cache=cache, rng=key)

        return sequence
