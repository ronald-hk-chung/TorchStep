# Implimentation of SSD300 with Resnet50 as backbone
# Ref: https://catalog.ngc.nvidia.com/orgs/nvidia/resources/ssd_for_pytorch
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet(nn.Module):
    """ResNet provide the feature provider backbone"""

    def __init__(self):
        super().__init__()
        # Loading Resent 50 backbone
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Enhancement from the Speed/accuracy trade-offs for modern convolutional object detectors paper
        # https://arxiv.org/abs/1611.10012
        # - The conv5_x, avgpool, fc and softmax layers were removed from the original classification model.
        # - All strides in conv4_x are set to 1x1.
        # Extract feature_provider to provide feature map [38x38] with 1024 channels
        self.feature_provider = nn.Sequential(*list(backbone.children())[:7])
        # change last feature_provider block conv and downsample to stride (1, 1)
        conv4_block = self.feature_provider[-1][0]
        conv4_block.conv1.stride = (1, 1)
        conv4_block.conv2.stride = (1, 1)
        conv4_block.downsample[0].stride = (1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # provide feature map in a forward pass
        x = self.feature_provider(x)
        return x  # [1024, 38, 38]


class SSD300(nn.Module):
    """SSD300 model with additional layers and classifcation and localisation heads"""

    def __init__(self, backbone=ResNet(), num_classes=91):  # number of COCO classes
        super().__init__()
        self.feature_provider = backbone  # initialise feature provider backbone
        self.num_classes = num_classes

        # dictionary mapping in_channels, out_channels and num_prior_box
        # need to build additional layer from 19x19 using
        features_map_dict = {
            "38x38": {"num_prior_box": 4, "feature_channels": 1024},
            "19x19": {
                "num_prior_box": 6,
                "intermediate_channels": 256,
                "feature_channels": 512,
                "padding": 1,
                "stride": 2,
            },
            "10x10": {
                "num_prior_box": 6,
                "intermediate_channels": 256,
                "feature_channels": 512,
                "padding": 1,
                "stride": 2,
            },
            "5x5": {
                "num_prior_box": 6,
                "intermediate_channels": 128,
                "feature_channels": 256,
                "padding": 1,
                "stride": 2,
            },
            "3x3": {
                "num_prior_box": 4,
                "intermediate_channels": 128,
                "feature_channels": 256,
                "padding": 0,
                "stride": 1,
            },
            "1x1": {
                "num_prior_box": 4,
                "intermediate_channels": 128,
                "feature_channels": 256,
                "padding": 0,
                "stride": 1,
            },
        }
        # Building additional features map from 19x19, 10x10, 5x5, 3x3 1x1
        features_list = list(features_map_dict.keys())
        self.additional_blocks = []
        for i, features_map in enumerate(features_list[1:]):
            # feature_channels of previous layer
            in_channels = features_map_dict[features_list[i]]["feature_channels"]
            # intermediate channels from conv1
            intermediate_channels = features_map_dict[features_map]["intermediate_channels"]
            # output_channel from conv2
            out_channels = features_map_dict[features_map]["feature_channels"]
            # padding for conv2, 1 for first 3 and 0 for last 2
            padding = features_map_dict[features_map]["padding"]
            # stride for conv2, 2 for first 3 and 1 for last 2
            stride = features_map_dict[features_map]["stride"]

            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=intermediate_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=intermediate_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=intermediate_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=padding,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True),
            )
            self.additional_blocks.append(layer)
        self.additional_blocks = nn.ModuleList(self.additional_blocks)

        # Generating localisation heads and classification heads
        self.loc = []
        self.conf = []
        for features_map in features_map_dict.values():
            self.loc.append(
                nn.Conv2d(
                    in_channels=features_map["feature_channels"],
                    out_channels=features_map["num_prior_box"] * 4,
                    kernel_size=3,
                    padding=1,
                )
            )
            self.conf.append(
                nn.Conv2d(
                    in_channels=features_map["feature_channels"],
                    out_channels=features_map["num_prior_box"] * self.num_classes,
                    kernel_size=3,
                    padding=1,
                )
            )
        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _init_weights(self):
        # Inialising on the newly created layers weights using xavier_uniform
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append(
                (
                    l(s).view(s.size(0), -1, 4),
                    c(s).view(s.size(0), -1, self.num_classes),
                )
            )

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 1).contiguous(), torch.cat(confs, 1).contiguous()
        return locs, confs

    def forward(self, x):
        x = torch.stack(x)  # stack up tuple of images into [batch, 3, 300, 300]
        x = self.feature_provider(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs
