import torch
import torch.nn as nn


class VGGNet(nn.Module):
    """VGGNet Implimentation
    Reference Paper: VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION
    Reference Link: https://arxiv.org/pdf/1409.1556v6.pdf
    """

    def __init__(
        self,
        vgg_cfg: str,
        in_channels: int = 3,
        batch_norm: bool = False,
        dropout: float = 0.5,
        num_classes: int = 1000,
        init_weights: bool = True,
    ):
        """Initialise VGGNet Class

        Args:
            vgg_cfg [str]: Configuration from original paper A-E
            in_channels [int]: num of color channels, Default 3
            batch_norm [bool]: Set true to use BatchNorm2d, Default False
            dropout [float]: probability of an element to be zeroed. Default 0.5
            num_classes [int]: num of classes to predict, Default 1000
            init_weight [bool]: initialise weights with kaiming_normal, Default True
        """
        super().__init__()

        # Initiatize cfgs for VGG Net including A-E
        # M - MaxPool2d with kernel_size=(2, 2) and stride=(2, 2)
        # tuple[k, v] - Conv2d with stride=1, padding=1, kernel_size=k and out_channels=v
        self.cfgs = {
            "A": [(3, 64), "M", (3, 128), "M", (3, 256), (3, 256), "M", (3, 512), (3, 512), "M", (3, 512), (3, 512), "M"],
            "B": [(3, 64), (3, 64), "M", (3, 128), (3, 128), "M", (3, 256), (3, 256), "M", (3, 512), (3, 512), "M", (3, 512), (3, 512), "M"],
            "C": [(3, 64), (3, 64), "M", (3, 128), (3, 128), "M", (3, 256), (3, 256), (1, 256), "M", (3, 512), (3, 512), (1, 512), "M", (3, 512), (3, 512), (1, 512), "M"],
            "D": [(3, 64), (3, 64), "M", (3, 128), (3, 128), "M", (3, 256), (3, 256), (3, 256), "M", (3, 512), (3, 512), (3, 512), "M", (3, 512), (3, 512), (3, 512), "M"],
            "E": [(3, 64), (3, 64), "M", (3, 128), (3, 128), "M", (3, 256), (3, 256), (3, 256), (3, 256), "M", (3, 512), (3, 512), (3, 512), (3, 512), "M", (3, 512), (3, 512), (3, 512), (3, 512), "M"],
        }

        # Make features layers depending on cfgs from vgg_cfg
        self.features = self.make_layers(vgg_cfg, in_channels, batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

        # Initialize neural network weights
        self.init_weights()

    def make_layers(self, vgg_cfg, in_channels, batch_norm):
        layers = nn.Sequential()
        # in_channels = self.in_channels
        for cfg in self.cfgs[vgg_cfg]:
            if cfg == "M":
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            else:
                kernel_size, out_channels = cfg
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=1,
                    )
                )
                if batch_norm:
                    layers.append(nn.BatchNorm2d(num_features=out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels

        return layers

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
