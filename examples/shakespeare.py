"""Trains a state space model on a character-level tokenized dataset of Shakespeare."""

from dataclasses import dataclass
from typing import Iterator, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
from jaxtyping import Array, PRNGKeyArray, PyTree

import xax


def cross_entropy(y: Array, pred_y: Array) -> Array:
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


@dataclass(frozen=True)
class ShakespeareDataset:
    text: str
    vocab: list[str]
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]


def load_shakespeare_text() -> ShakespeareDataset:
    """Loads the Tiny Shakespeare dataset.

    This function loads the tiny_shakespeare dataset from tfds, extracts the
    text, and builds a character-level tokenizer.

    Returns:
        The loaded dataset.
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
    return ShakespeareDataset(text, vocab, token_to_id, id_to_token)


@dataclass
class Config(xax.Config):
    input_size: int = xax.field(65)
    output_size: int = xax.field(65)
    num_layers: int = xax.field(3)
    hidden_size: int = xax.field(512)
    batch_size: int = xax.field(128)
    learning_rate: float = xax.field(1e-3)
    sequence_length: int = xax.field(100)
    valid_every_n_seconds: float = xax.field(30.0)
    model_type: str = xax.field("s4", help="The model to use")


class RecurrentModel(Protocol):
    def predict_sequence(self, x_seq: Array) -> Array: ...


class RNN(eqx.Module):
    vocab_embedding: eqx.nn.Embedding
    rnn_cells: list[eqx.nn.GRUCell]
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        vocab_key, rnn_key = jax.random.split(key, 2)
        self.vocab_embedding = eqx.nn.Embedding(input_size, hidden_size, key=vocab_key)
        keys = jax.random.split(rnn_key, num_layers)
        self.rnn_cells = [
            eqx.nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size, key=keys[i]) for i in range(num_layers)
        ]
        self.output_layer = eqx.nn.Linear(hidden_size, output_size, key=keys[-1])

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


class LSTM(eqx.Module):
    vocab_embedding: eqx.nn.Embedding
    rnn_cells: list[eqx.nn.LSTMCell]
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        vocab_key, rnn_key = jax.random.split(key, 2)
        self.vocab_embedding = eqx.nn.Embedding(input_size, hidden_size, key=vocab_key)
        keys = jax.random.split(rnn_key, num_layers)
        self.rnn_cells = [
            eqx.nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size, key=keys[i]) for i in range(num_layers)
        ]
        self.output_layer = eqx.nn.Linear(hidden_size, output_size, key=keys[-1])

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


class ShakespearePrediction(xax.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.ds = load_shakespeare_text()
        self.token_ids = jnp.array([self.ds.token_to_id[c] for c in self.ds.text], dtype=jnp.int32)

    def compute_metrics(
        self,
        model: PyTree,
        batch: tuple[Array, Array],
        output: Array,
        loss: Array,
        state: xax.State,
    ) -> dict[str, Array]:
        _, y = batch
        yhat = output.argmax(axis=-1)
        return {
            "loss": loss,
            "acc": (yhat == y).astype(float).mean(),
        }

    def get_model(self, key: PRNGKeyArray) -> RecurrentModel:
        match self.config.model_type:
            case "rnn":
                return RNN(
                    input_size=self.config.input_size,
                    hidden_size=self.config.hidden_size,
                    output_size=self.config.output_size,
                    num_layers=self.config.num_layers,
                    key=key,
                )
            case "lstm":
                return LSTM(
                    input_size=self.config.input_size,
                    hidden_size=self.config.hidden_size,
                    output_size=self.config.output_size,
                    num_layers=self.config.num_layers,
                    key=key,
                )
            case "s4":
                return xax.S4(
                    input_size=self.config.input_size,
                    hidden_size=self.config.hidden_size,
                    output_size=self.config.output_size,
                    num_layers=self.config.num_layers,
                    block_type="diag",
                    skip_connections=True,
                    key=key,
                )
            case _:
                raise ValueError(f"Unknown model type: {self.config.model_type}")

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adam(self.config.learning_rate)

    def get_output(self, model: RecurrentModel, batch: tuple[Array, Array], state: xax.State) -> Array:
        x_batched, _ = batch
        return jax.vmap(model.predict_sequence)(x_batched)

    def compute_loss(self, model: RecurrentModel, batch: tuple[Array, Array], output: Array, state: xax.State) -> Array:
        (_, y), yhat = batch, output
        return xax.cross_entropy(y, yhat, axis=-1).mean()

    def log_valid_step(
        self,
        batch: tuple[Array, Array],
        output: Array,
        metrics: xax.FrozenDict[str, Array],
        state: xax.State,
    ) -> None:
        output_tokens = jnp.argmax(output, axis=-1)[0]
        output_words = "".join([self.ds.id_to_token[int(token)] for token in output_tokens])
        self.logger.log_string("output", output_words)

    def get_data_iterator(self, phase: xax.Phase) -> Iterator[tuple[Array, Array]]:
        """Returns an iterator over batches of tokenized Shakespeare text.

        Args:
            phase: The phase of the data iterator to return.

        Returns:
            An iterator over batches of tokenized Shakespeare text, with
            each batch consisting of a tuple of the input tokens and the
            target tokens (shifted by one position).
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
