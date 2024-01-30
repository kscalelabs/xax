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

    def get_dataset(self, phase: xax.Phase) -> xax.MNIST:
        return xax.MNIST(train=phase == "train")


if __name__ == "__main__":
    # python -m examples.mnist
    MnistClassification.launch(Config(batch_size=16))
