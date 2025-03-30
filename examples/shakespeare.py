"""Trains a state space model on a character-level tokenized dataset of Shakespeare."""

from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, replace
from typing import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds  # for loading tiny_shakespeare
from jaxtyping import Array, PRNGKeyArray

import xax

# ------------------------------------------------------------------------------
# Tokenizer and Data Loading for Tiny Shakespeare
# ------------------------------------------------------------------------------


def load_shakespeare_text() -> tuple[str, list[str], dict[str, int], dict[int, str]]:
    """
    Loads the tiny_shakespeare dataset from tfds, extracts the text,
    and builds a character-level tokenizer.
    """
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    # tiny_shakespeare consists of a single example with the full text.
    for example in tfds.as_numpy(ds):
        # the text field might be bytes, so decode if necessary.
        text = example["text"]
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        break

    # Build vocabulary from unique characters in the text.
    vocab = sorted(list(set(text)))
    token_to_id = {ch: i for i, ch in enumerate(vocab)}
    id_to_token = {i: ch for i, ch in enumerate(vocab)}
    return text, vocab, token_to_id, id_to_token


# ------------------------------------------------------------------------------
# Config and Model Definitions (unchanged)
# ------------------------------------------------------------------------------


@dataclass
class Config(xax.Config):
    input_size: int = xax.field(65)
    output_size: int = xax.field(65)
    num_layers: int = xax.field(3)
    hidden_size: int = xax.field(512)
    # Training parameters.
    batch_size: int = xax.field(128)
    learning_rate: float = xax.field(1e-3)
    sequence_length: int = xax.field(100)
    valid_every_n_seconds: float = xax.field(30.0)


class RecurrentModel(ABC):
    @abstractmethod
    def predict_sequence(self, x_seq: Array) -> Array:
        pass


class RNN(eqx.Module, RecurrentModel):
    vocab_embedding: eqx.nn.Embedding
    rnn_cells: list[eqx.nn.GRUCell]
    output_layer: eqx.nn.Linear

    def __init__(self, config: Config, *, key: PRNGKeyArray):
        vocab_key, rnn_key = jax.random.split(key, 2)
        self.vocab_embedding = eqx.nn.Embedding(config.input_size, config.hidden_size, key=vocab_key)
        keys = jax.random.split(rnn_key, config.num_layers)
        self.rnn_cells = [
            eqx.nn.GRUCell(input_size=config.hidden_size, hidden_size=config.hidden_size, key=keys[i])
            for i in range(config.num_layers)
        ]
        self.output_layer = eqx.nn.Linear(config.hidden_size, config.output_size, key=keys[-1])

    def __call__(self, hs: list[Array], x: Array) -> tuple[list[Array], Array]:
        new_hs = []
        for i, rnn_cell in enumerate(self.rnn_cells):
            h = rnn_cell(x, hs[i])
            new_hs.append(h)
            x = h  # Pass the output of the current layer as input to the next
        y = self.output_layer(x)
        return new_hs, y

    def predict_sequence(self, x_seq: Array) -> Array:
        hs = [jnp.zeros(cell.hidden_size) for cell in self.rnn_cells]
        x_seq = jax.vmap(self.vocab_embedding)(x_seq)

        def step(hs: list[Array], x: Array) -> tuple[list[Array], Array]:
            hs, y = self(hs, x)
            return hs, y

        _, y_seq = jax.lax.scan(step, hs, x_seq)
        return y_seq


class LSTM(eqx.Module, RecurrentModel):
    vocab_embedding: eqx.nn.Embedding
    rnn_cells: list[eqx.nn.LSTMCell]
    output_layer: eqx.nn.Linear

    def __init__(self, config: Config, *, key: PRNGKeyArray):
        vocab_key, rnn_key = jax.random.split(key, 2)
        self.vocab_embedding = eqx.nn.Embedding(config.input_size, config.hidden_size, key=vocab_key)
        keys = jax.random.split(rnn_key, config.num_layers)
        self.rnn_cells = [
            eqx.nn.LSTMCell(input_size=config.hidden_size, hidden_size=config.hidden_size, key=keys[i])
            for i in range(config.num_layers)
        ]
        self.output_layer = eqx.nn.Linear(config.hidden_size, config.output_size, key=keys[-1])

    def __call__(self, hs: list[tuple[Array, Array]], x: Array) -> tuple[list[tuple[Array, Array]], Array]:
        new_hs: list[tuple[Array, Array]] = []
        for i, rnn_cell in enumerate(self.rnn_cells):
            h, c = rnn_cell(x, hs[i])
            new_hs.append((h, c))
            x = h  # Pass the output of the current layer as input to the next
        y = self.output_layer(x)
        return new_hs, y

    def predict_sequence(self, x_seq: Array) -> Array:
        hs = [(jnp.zeros(cell.hidden_size), jnp.zeros(cell.hidden_size)) for cell in self.rnn_cells]
        x_seq = jax.vmap(self.vocab_embedding)(x_seq)

        def step(hs: list[tuple[Array, Array]], x: Array) -> tuple[list[tuple[Array, Array]], Array]:
            hs, y = self(hs, x)
            return hs, y

        _, y_seq = jax.lax.scan(step, hs, x_seq)
        return y_seq


class SSMBlock(eqx.Module):
    A: Array
    B: Array

    def __init__(self, config: Config, *, key: PRNGKeyArray):
        glorot = lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0) * jnp.sqrt(2 / sum(shape))

        key_A, key_B = jax.random.split(key)
        self.A = glorot(key_A, (config.hidden_size, config.hidden_size))
        self.B = glorot(key_B, (config.hidden_size, config.hidden_size))

    def forward(self, h: Array, x: Array) -> Array:
        h = self.A @ h + self.B.T @ x
        return h

    def get_kernel(self, L: int) -> Array:
        return self.A


class DiagSSMBlock(eqx.Module):
    a: Array
    B: Array

    def __init__(self, config: Config, *, key: PRNGKeyArray):
        keys = jax.random.split(key, 2)
        glorot = lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0) * jnp.sqrt(2 / sum(shape))
        self.a = glorot(keys[0], (config.hidden_size,))
        self.B = glorot(keys[1], (config.hidden_size, config.hidden_size))

    def forward(self, h: Array, x: Array) -> Array:
        h = self.a * h + self.B.T @ x
        h = jax.nn.tanh(h)
        return h

    def get_kernel(self, L: int) -> Array:
        """Returns the kernel with time as the final dimension."""
        exponents = jnp.arange(L)
        kernel = jnp.power(self.a[:, None], exponents)  # (H, L)
        kernel = kernel[:, None, :]  # (H, 1, L)
        return kernel

    def forward_accross_time(self, x: Array) -> Array:
        """Convolves x (T, H) across time using the kernel."""
        T, H = x.shape

        # Compute s = x @ U.T + b, with shape (N, T, H)
        s = self.B.T @ x
        s = s.T  # (H, T)

        kernel = self.get_kernel(T)  # (H, 1, T)
        kernel_flipped = jnp.flip(kernel, axis=-1)

        # Pad s on the left along the time axis (pad length T-1)
        s_padded = jnp.pad(s, ((0, 0), (0, 0), (T - 1, 0)))

        # Perform depthwise (grouped) 1D convolution.
        # We use input shape (N, H, L) and kernel shape (H, 1, T) with feature_group_count=H.
        # The dimension_numbers are chosen so that the channel dimension is second.
        conv_out = jax.lax.conv_general_dilated(
            s_padded,
            kernel_flipped,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCH", "OIH", "NCH"),
            feature_group_count=H,
        )
        # conv_out has shape (N, H, T); transpose to (N, T, H)
        conv_out = jnp.transpose(conv_out, (0, 2, 1))
        return conv_out

    def naive_forward_accross_time(self, x: Array) -> Array:
        """Naively forward across time."""

        def step(h: Array, x: Array) -> tuple[Array, Array]:
            h = self.forward(h, x)
            return h, h

        h_0 = jnp.zeros(self.a.shape[0])
        _, h_seq = jax.lax.scan(step, h_0, x)
        return h_seq


class DPLRSSMBlock(eqx.Module):
    d: Array  # Diagonal component, analogous to self.a in DiagSSMBlock
    L: Array  # Left low-rank factor
    R: Array  # Right low-rank factor
    B: Array  # Input transformation matrix

    def __init__(self, config: Config, *, key: PRNGKeyArray):
        self.d = jax.nn.initializers.glorot_uniform()(key, (config.hidden_size,))
        rank = 4
        self.L = jax.nn.initializers.glorot_uniform()(key, (config.hidden_size, rank))
        self.R = jax.nn.initializers.glorot_uniform()(key, (rank, config.hidden_size))
        # Input transformation matrix.
        self.B = jax.nn.initializers.glorot_uniform()(key, (config.hidden_size, config.hidden_size))

    def forward(self, h: Array, x: Array) -> Array:
        low_rank_update = self.L @ (self.R @ h)
        h = self.d * h + low_rank_update + self.B.T @ x
        h = jax.nn.tanh(h)
        return h


class S4(eqx.Module, RecurrentModel):
    vocab_embedding: eqx.nn.Embedding
    proj_in: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    blocks: list[DPLRSSMBlock | DiagSSMBlock | SSMBlock]
    num_layers: int = eqx.static_field()
    hidden_size: int = eqx.static_field()

    def __init__(self, config: Config, *, key: PRNGKeyArray):
        vocab_key, s4_key = jax.random.split(key, 2)
        self.vocab_embedding = eqx.nn.Embedding(config.input_size, config.hidden_size, key=vocab_key)
        self.proj_in = eqx.nn.Linear(config.hidden_size, config.hidden_size, key=key)
        self.proj_out = eqx.nn.Linear(config.hidden_size, config.output_size, key=key)

        block_keys = jax.random.split(s4_key, config.num_layers)
        # self.blocks = [DiagSSMBlock(config, key=block_keys[i]) for i in range(config.num_layers)]
        # self.blocks = [DPLRSSMBlock(config, key=block_keys[i]) for i in range(config.num_layers)]
        self.blocks = [SSMBlock(config, key=block_keys[i]) for i in range(config.num_layers)]

        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size

    def __call__(self, hs: list[Array], x: Array) -> tuple[list[Array], Array]:
        new_hs = []
        for i, block in enumerate(self.blocks):
            h = block.forward(hs[i], x)
            new_hs.append(h)
            # TODO: maybe add skip connection
            x = jax.nn.gelu(h)

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


# ------------------------------------------------------------------------------
# Shakespeare Prediction Task using Tiny Shakespeare and a character-level tokenizer
# ------------------------------------------------------------------------------


class ShakespearePrediction(xax.Task[Config]):
    def __init__(self, config: Config):
        self.text, self.vocab, self.token_to_id, self.id_to_token = load_shakespeare_text()
        self.token_ids = jnp.array([self.token_to_id[c] for c in self.text], dtype=jnp.int32)
        super().__init__(config)

    def get_model(self, key: PRNGKeyArray) -> RecurrentModel:
        # For example, using the S4 model.
        # return S4(self.config, key=key)
        # return RNN(self.config, key=key)
        return LSTM(self.config, key=key)

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adam(self.config.learning_rate)

    def get_output(self, model: RecurrentModel, batch: tuple[Array, Array]) -> Array:
        x_batched, _ = batch
        return jax.vmap(model.predict_sequence)(x_batched)

    def compute_loss(self, model: RecurrentModel, batch: tuple[Array, Array], output: Array) -> Array:
        # Target y is of shape (B, T) with integer token IDs.
        _, y = batch
        # Convert targets to one-hot.
        one_hot_y = jax.nn.one_hot(y, self.config.output_size)
        # Compute cross-entropy loss over the sequence.
        loss = optax.softmax_cross_entropy(logits=output, labels=one_hot_y).mean()
        return loss

    def log_train_step(
        self, model: RecurrentModel, batch: tuple[Array, Array], output: Array, state: xax.State
    ) -> None:
        loss = self.compute_loss(model, batch, output)
        self.logger.log_scalar("train_loss", loss)

    def log_valid_step(
        self, model: RecurrentModel, batch: tuple[Array, Array], output: Array, state: xax.State
    ) -> None:
        loss = self.compute_loss(model, batch, output)
        self.logger.log_scalar("valid_loss", loss)

        # logging the first sequence of the output.
        output_tokens = jnp.argmax(output, axis=-1)[0]
        output_words = "".join([self.id_to_token[int(token)] for token in output_tokens])
        self.logger.log_string("output", output_words)

    def get_data_iterator(self, phase: xax.Phase) -> Iterator:
        """
        Returns an iterator over batches of tokenized Shakespeare text.
        Each batch consists of a tuple:
          - inputs: one-hot encoded tokens of shape (B, T, vocab_size)
          - targets: integer token IDs of shape (B, T)
        """
        seq_len = self.config.sequence_length
        # Split the token_ids into training and validation sets.
        if phase == "train":
            token_ids = self.token_ids[: int(0.8 * len(self.token_ids))]
        else:
            token_ids = self.token_ids[int(0.8 * len(self.token_ids)) :]
        n_tokens = token_ids.shape[0]

        key = jax.random.PRNGKey(0)
        while True:
            key, subkey = jax.random.split(key)
            # Sample starting indices for each sequence in the batch.
            idx = jax.random.randint(subkey, (self.config.batch_size,), 0, n_tokens - seq_len - 1)
            # Build the batch: for each starting index, extract a sequence of length seq_len + 1.
            batch_x = jnp.stack([token_ids[i : i + seq_len] for i in idx])
            batch_y = jnp.stack([token_ids[i + 1 : i + seq_len + 1] for i in idx])
            # One-hot encode the input sequences.
            yield batch_x, batch_y


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Launch the training task.
    ShakespearePrediction.launch(Config())
