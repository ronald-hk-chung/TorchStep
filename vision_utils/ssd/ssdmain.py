import torch.nn as nn
from .multibox import MultiBoxLoss
from ssd_resnetbackbone import SSD300

class SSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = SSD300(num_classes=num_classes)
        self.mbl = MultiBoxLoss(num_classes=num_classes)
    def forward(self, *args):
        if self.training:
            input = args[0]
            targets = args[1]
            preds = self.backbone(input)
            losses = self.mbl(preds, targets)
            return losses
        else:
            input = args[0]
            preds = self.backbone(input)
            return preds