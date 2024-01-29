"""Trains a simple convolutional neural network on the MNIST dataset.

Run this example with `python -m examples.mnist`.
"""

from dataclasses import dataclass

import flax.linen as nn

import xax


@dataclass
class Config(xax.Config):
    in_dim: int = xax.field(1, help="Number of input dimensions")


class MnistClassification(xax.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.model = nn.Sequential(
            [
                nn.Conv(config.in_dim, 3, padding=1),
                nn.BatchNorm(),
                nn.relu,
                nn.Conv(32, 3, padding=1),
                nn.BatchNorm(),
                nn.relu,
                nn.Conv(64, 3, padding=1),
                nn.BatchNorm(),
                nn.relu,
                nn.Conv(64, 3, padding=1),
                nn.BatchNorm(),
                nn.relu,
            ],
        )

    def get_dataset(self, phase: xax.Phase) -> Dataset[tuple[Array, Array]]:
        root_dir = xax.get_data_dir() / "mnist"
        return MNIST(root=root_dir, train=phase == "train", download=not root_dir.exists())

    def build_optimizer(self) -> Optimizer:
        return optax.adam

    def get_loss(self, batch: tuple[Tensor, Tensor], state: mlfab.State) -> Tensor:
        x, y = batch
        yhat = self(x)
        self.log_step(batch, yhat, state)
        return F.cross_entropy(yhat, y.squeeze(-1).long())

    def log_valid_step(self, batch: tuple[Tensor, Tensor], output: Tensor, state: mlfab.State) -> None:
        (x, y), yhat = batch, output

        def get_label_strings() -> list[str]:
            ytrue, ypred = y.squeeze(-1), yhat.argmax(-1)
            return [f"ytrue={ytrue[i]}, ypred={ypred[i]}" for i in range(len(ytrue))]

        self.log_labeled_images("images", lambda: (x, get_label_strings()))


if __name__ == "__main__":
    MnistClassification.launch(Config(batch_size=16))
