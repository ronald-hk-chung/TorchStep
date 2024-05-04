from torch import nn
from collections import OrderedDict


class LeNet5(nn.Module):
    """LeNet-5 implimentation
    Referece: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf?ref=blog.paperspace.com
    """

    def __init__(self, in_features: int, out_features: int):
        """Initialize the LeNet Class

        Args:
            in_features (int): no. of color channels
            out_features (int): no. of classes in final classifier dense layer
        """
        super().__init__()
        self.feature = nn.Sequential(
            OrderedDict(
                [
                    # Block 1, 1@28x28 -> 6@28x28 -> 6@14x14
                    # Note that we are using padding=2 to add padding to each side of 28x28 -> 32x32
                    (
                        "C1",
                        nn.Conv2d(
                            in_channels=in_features,
                            out_channels=6,
                            kernel_size=5,
                            padding=2,
                        ),
                    ),
                    ("func1", nn.ReLU()),
                    ("S2", nn.MaxPool2d(kernel_size=2)),
                    # Block 2, 6@14x14,-> 16@10x10,-> 16@5x5
                    ("C3", nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)),
                    ("func2", nn.ReLU()),
                    ("S4", nn.MaxPool2d(kernel_size=2)),
                    # Block 3, 16@5x5,-> 120@1x1
                    ("C5", nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)),
                    ("func3", nn.ReLU()),
                    ("flatten", nn.Flatten()),
                ]
            )
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("F6", nn.Linear(in_features=120, out_features=84)),
                    ("func4", nn.ReLU()),
                    ("output", nn.Linear(in_features=84, out_features=out_features)),
                ]
            )
        )

    def forward(self, X):
        return self.classifier(self.feature(X))
