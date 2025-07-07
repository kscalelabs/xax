# mypy: disable-error-code="import-not-found"
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
    num_layers: int = xax.field(4)
    hidden_size: int = xax.field(256)
    batch_size: int = xax.field(12)
    learning_rate: float = xax.field(1e-3)
    sequence_length: int = xax.field(1024)
    valid_every_n_seconds: float = xax.field(30.0)
    model_type: str = xax.field("lstm", help="The model to use")


class SequenceModel(Protocol):
    def predict_sequence(self, x_seq: Array) -> Array: ...

    def generate_sequence(self, prompt_seq: Array, max_len: int) -> Array: ...


class RNN(eqx.Module):
    vocab_embedding: eqx.nn.Embedding
    rnn_cells: list[eqx.nn.GRUCell]
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
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

    def generate_sequence(self, prompt_seq: Array, max_len: int) -> Array:
        hs = [jnp.zeros(cell.hidden_size) for cell in self.rnn_cells]
        prompt_seq_embedded = jax.vmap(self.vocab_embedding)(prompt_seq)

        def encode_step(hs: list[Array], x: Array) -> tuple[list[Array], Array]:
            hs, y = self(hs, x)
            return hs, y

        def decode_step(
            carry: tuple[list[Array], Array, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[list[Array], Array, PRNGKeyArray], Array]:
            hs, last_token, rng = carry
            token_embedded = self.vocab_embedding(last_token)
            hs, y = self(hs, token_embedded)
            token = jax.random.categorical(rng, y)
            rng = jax.random.split(rng)[0]
            return (hs, token, rng), token

        hs, _ = jax.lax.scan(encode_step, hs, prompt_seq_embedded)
        _, sequence = jax.lax.scan(decode_step, (hs, prompt_seq[-1], jax.random.PRNGKey(0)), None, length=max_len)

        return sequence


class LSTM(eqx.Module):
    vocab_embedding: eqx.nn.Embedding
    rnn_cells: list[eqx.nn.LSTMCell]
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
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

    def generate_sequence(self, prompt_seq: Array, max_len: int) -> Array:
        hs = [(jnp.zeros(cell.hidden_size), jnp.zeros(cell.hidden_size)) for cell in self.rnn_cells]
        prompt_seq_embedded = jax.vmap(self.vocab_embedding)(prompt_seq)

        def encode_step(hs: list[tuple[Array, Array]], x: Array) -> tuple[list[tuple[Array, Array]], Array]:
            hs, y = self(hs, x)
            return hs, y

        def decode_step(
            carry: tuple[list[tuple[Array, Array]], Array, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[list[tuple[Array, Array]], Array, PRNGKeyArray], Array]:
            hs, last_token, rng = carry
            token_embedded = self.vocab_embedding(last_token)
            hs, y = self(hs, token_embedded)
            token = jax.random.categorical(rng, y)
            rng = jax.random.split(rng)[0]
            return (hs, token, rng), token

        hs, _ = jax.lax.scan(encode_step, hs, prompt_seq_embedded)
        _, sequence = jax.lax.scan(decode_step, (hs, prompt_seq[-1], jax.random.PRNGKey(0)), None, length=max_len)

        return sequence


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

    def get_model(self, key: PRNGKeyArray) -> SequenceModel:
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
            case "ssm":
                return xax.SSM(
                    input_size=self.config.input_size,
                    hidden_size=self.config.hidden_size,
                    output_size=self.config.output_size,
                    num_layers=self.config.num_layers,
                    block_type="diagonal",
                    skip_connections=True,
                    discretize=False,
                    key=key,
                )
            case "transformer":
                return xax.Transformer(
                    vocab_size=self.config.input_size,
                    embed_dim=self.config.hidden_size,
                    num_heads=self.config.hidden_size // 64,
                    ff_dim=self.config.hidden_size * 4,
                    num_layers=self.config.num_layers,
                    output_size=self.config.output_size,
                    key=key,
                )
            case _:
                raise ValueError(f"Unknown model type: {self.config.model_type}")

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adamw(
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
        )

    def get_output(self, model: SequenceModel, batch: tuple[Array, Array], state: xax.State) -> Array:
        x_batched, _ = batch
        return jax.vmap(model.predict_sequence)(x_batched)

    def compute_loss(self, model: SequenceModel, batch: tuple[Array, Array], output: Array, state: xax.State) -> Array:
        (_, y), yhat = batch, output
        labels = jax.nn.one_hot(y, yhat.shape[-1])
        return optax.softmax_cross_entropy(logits=yhat, labels=labels).mean()

    def log_valid_step(
        self,
        model: SequenceModel,
        batch: tuple[Array, Array],
        output: Array,
        metrics: xax.FrozenDict[str, Array],
        state: xax.State,
    ) -> None:
        output_tokens = jnp.argmax(output, axis=-1)[0]
        output_words = "".join([self.ds.id_to_token[int(token)] for token in output_tokens])
        self.logger.log_string("teacher_forced_output", output_words)

        # Using the first few tokens from the batch, generate the rest of the sequence.
        prompt_seq = jnp.array([self.ds.token_to_id[c] for c in "To be"])
        generated_tokens = model.generate_sequence(prompt_seq, max_len=500)
        generated_words = "".join([self.ds.id_to_token[int(token)] for token in generated_tokens])
        self.logger.log_string("prompt", "".join([self.ds.id_to_token[int(token)] for token in prompt_seq]))
        self.logger.log_string("generated_output", generated_words)

    def get_data_iterator(self, phase: xax.Phase, key: PRNGKeyArray) -> Iterator[tuple[Array, Array]]:
        """Returns an iterator over batches of tokenized Shakespeare text.

        Args:
            phase: The phase of the data iterator to return.
            key: The PRNG key to use for the data iterator.

        Returns:
            An iterator over batches of tokenized Shakespeare text, with
            each batch consisting of a tuple of the input tokens and the
            target tokens (shifted by one position).
        """
        seq_len = self.config.sequence_length
        # Split the token_ids into training and validation sets.
        if phase == "train":
            token_ids = self.token_ids[: int(0.95 * len(self.token_ids))]
        else:
            token_ids = self.token_ids[int(0.95 * len(self.token_ids)) :]
        n_tokens = token_ids.shape[0]

        while True:
            key, subkey = jax.random.split(key)
            # Sample starting indices for each sequence in the batch.
            idx = jax.random.randint(subkey, (self.config.batch_size,), 0, n_tokens - seq_len - 1)
            # Build the batch: for each starting index, extract a sequence of length seq_len + 1.
            batch_x = jnp.stack([token_ids[i : i + seq_len] for i in idx])
            batch_y = jnp.stack([token_ids[i + 1 : i + seq_len + 1] for i in idx])
            # One-hot encode the input sequences.
            yield batch_x, batch_y


if __name__ == "__main__":
    # Launch the training task.
    #   python -m examples.shakespeare
    ShakespearePrediction.launch(
        Config(
            model_type="ssm",
        )
    )
