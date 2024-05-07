import torch.nn as nn
from collections import OrderedDict

class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()

        # 5 conv block for features
        self.conv_block1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=(11, 11), stride=4, padding=2)),
            ('relu1', nn.ReLU()),
            ('lrn1', nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)),
            ('maxpool1', nn.MaxPool2d(kernel_size=(3, 3), stride=2))
        ]))

        self.conv_block2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=2)),
            ('relu2', nn.ReLU()),
            ('lrn2', nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)),
            ('maxpool2', nn.MaxPool2d(kernel_size=(3, 3), stride=2))
        ]))

        self.conv_block3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1)),
            ('relu3', nn.ReLU())
        ]))

        self.conv_block4 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1)),
            ('relu4', nn.ReLU())
        ]))

        self.conv_block5 = nn.Sequential(OrderedDict([
            ('conv5', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)),
            ('relu5', nn.ReLU()),
            ('maxpool5', nn.MaxPool2d(kernel_size=(3, 3), stride=2))
        ]))

        self.features = nn.Sequential(OrderedDict([
            ('conv_block1', self.conv_block1),
            ('conv_block2', self.conv_block2),
            ('conv_block3', self.conv_block3),
            ('conv_block4', self.conv_block4),
            ('conv_block5', self.conv_block5)
        ]))

        # Avg pool layer before classifier
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))

        # Define fc for classifiers
        self.fc1 = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten()),
            ('linear1', nn.Linear(in_features=9216, out_features=4096)),
            ('dropout1', nn.Dropout(p=0.5)),
            ('relu_fc1', nn.ReLU())
        ]))

        self.fc2 = nn.Sequential(OrderedDict([
            ('linear2', nn.Linear(in_features=4096, out_features=4096)),
            ('dropout2', nn.Dropout(p=0.5)),
            ('relu_fc2', nn.ReLU())
        ]))

        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
        self.classifier = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x