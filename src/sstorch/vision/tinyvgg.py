import torch.nn as nn
from collections import OrderedDict


class TinyVGG(nn.Module):
    """A simple Convolutional Neural Network built with 2 Conv Block
    Conv Block consists
    1. Convolutional Layer (nn.Conv2d)
    2. Activation Function (nn.ReLU)
    3. Pooling Layer (nn.MaxPool2d)
    Referece: https://poloclub.github.io/cnn-explainer/
    """

    def __init__(self, in_channels: int, hidden_units: int, out_channels: int):
        """Initialize the Convolutional Layers

        Args:
            in_channels (int): no. of color channels
            hidden_units (int): no. of features in hidden layers
            out_channels (int): no. of classes in final classifier dense layer

        """
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=hidden_units,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu1", nn.ReLU()),
                    (
                        "conv2",
                        nn.Conv2d(
                            in_channels=hidden_units,
                            out_channels=hidden_units,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu2", nn.ReLU()),
                    ("maxpool", nn.MaxPool2d(kernel_size=2)),
                ]
            )
        )
        self.conv_block_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            in_channels=hidden_units,
                            out_channels=hidden_units,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu1", nn.ReLU()),
                    (
                        "conv2",
                        nn.Conv2d(
                            in_channels=hidden_units,
                            out_channels=hidden_units,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu2", nn.ReLU()),
                    ("maxpool", nn.MaxPool2d(kernel_size=2)),
                ]
            )
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("flattener", nn.Flatten()),
                    ("linear", nn.LazyLinear(out_features=out_channels)),
                ]
            )
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
