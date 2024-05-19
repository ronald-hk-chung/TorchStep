from typing import List, Optional, Type, Union

import torch
from torch import Tensor
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        """BasicBlock Class for ResNet-18 and ResNet-34

        Args:
            inplanes (int): num of input dimensions of BasicBlock
            planes (int): num of output dimensions of BasicBlock
            stride (int): stride use in first 3x3 convolutional layer
            downsample (nn.Module): downsample input identity in case stride!=1 for the shape to match
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, 
                               out_channels=planes, 
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=planes,
                               out_channels=planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None):
        """Bottneck for ResNet-50, ResNet-101 and ResNet-152

        Args:
            inplanes (int): num of input dimensions of Botteneck
            planes (int): num of output dimensions for intermediate first 1x1 and second 3x3 convolution
            stride (int): stride use in 3x3 convolutional layer
            downsample (nn.Module): downsample input identity in case stride!=1 for the shape to match
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, 
                               out_channels=planes,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.conv2 = nn.Conv2d(in_channels=planes, 
                               out_channels=planes, 
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.conv3 = nn.Conv2d(in_channels=planes, 
                               out_channels=planes * self.expansion,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000):
        """Simplified ResNet Implimentation
        
        Args:
            block (nn.Module): residual block type
            layers (list[int]): num of esidual blocks for each conv layer
            num_classes: number of classes for image classification, Default 1000
        
        Reference for ResNet Architecture
        ResNet-18: ResNet(BasicBlock, [2, 2, 2, 2], 1000)
        ResNet-34: ResNet(BasicBlock, [3, 4, 6, 3], 1000)
        ResNet-50: ResNet(Bottleneck, [3, 4, 6, 3], 1000)
        ResNet-101: ResNet(Bottleneck, [3, 4, 23, 3], 1000)
        ResNet-152: ResNet(Bottleneck, [3, 8, 36, 3], 1000)
        """
        super().__init__()

        # Resnet stem to downsample image to speed up training
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=self.inplanes, 
                               kernel_size=7, 
                               stride=2, 
                               padding=3, 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, 
                                    stride=2, 
                                    padding=1)

        # Residual Blocks conv2_x to conv5_x
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization using kaiming_normal
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1):

        # downsample for identity with same stride if stride!=1
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplanes, 
                          out_channels=planes * block.expansion, 
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(num_features=planes * block.expansion),
            )
        # initiailise layer
        # append first layer with downsample with same stride if stride!=1
        layers = []
        layers.append(block(inplanes=self.inplanes,
                            planes=planes,
                            stride=stride,
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes,
                                planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x