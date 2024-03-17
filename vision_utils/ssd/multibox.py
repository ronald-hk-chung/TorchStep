import torch
import torch.nn.functional as F
import torch.nn as nn
from .prior import PriorBox
from .utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Reference: https://arxiv.org/pdf/1512.02325.pdf

    Workflow:
    1)  Produce Confidence Target Indices by matching ground truth boxes
        with (default) priorboxes that have jaccard index > threshold parameter
    2)  Produce localisation target by 'encoding variance into offsets of groud
        truth boxes and their matched 'priorboxes'
    3)  Hard negative mining to filter the excessive number of negative examples
        that comes with using a large number of default bounding boxes (default: neg:pos=3:1)

    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where:
            Lconf is the CrossEntropy Loss
            Lloc is the SmoothL1 Loss
            weighted by α which is set to 1 by cross val.

            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes

    """

    def __init__(self, iou_thresold=0.5, neg_pos=3, num_classes=91):
        super().__init__()
        self.num_classes = num_classes
        self.thresold = iou_thresold
        self.negpos_ratio = neg_pos
        self.variance = [0.1, 0.2]
        # Generate and assign priors to buffer
        priors = PriorBox()()  # priors shape: torch.size(num_priors, 4) in CXCYWH
        # not to be considered a model parameter and trained by optimizer
        self.register_buffer("priors", priors)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, preds, targets):
        """Multibox Loss
        Args:
            preds (tuple[tensor]):  (loc_data, conf_data)
                loc_data:   Shape: (batch_size, num_priors, 4)
                conf_data:  Shape: (batch_size, num_priors, num_classes)

            targets (tensor):   Ground truth boxes and labels for a batch
                Shape: (batch_size, num_objs, 5) (last idx is the label)

        Returns:
            dict[tensor]: dictionary of tensors including
                loss_l (tensor): smooth_l1_loss of predicted location and groundtruth location
                loss_c (tensor): cross_entropy loss of predicted classification and groundtruth classification
        """
        loc_data, conf_data = preds
        batch_size = loc_data.size(0)  # batch size
        num_priors = self.priors.size(0)  # num_priors, default 8732

        # match priors and ground truth boxes
        loc_t = torch.zeros(
            size=(batch_size, num_priors, 4),
            dtype=torch.float,
            requires_grad=False,
            device=self.device,
        )
        conf_t = torch.zeros(
            size=(batch_size, num_priors),
            dtype=torch.long,
            requires_grad=False,
            device=self.device,
        )

        for batch in range(batch_size):
            boxes = targets[batch]["boxes"] / 300  # in XYXY
            labels = targets[batch]["labels"]
            loc_t[batch], conf_t[batch] = match(
                self.thresold, boxes, self.priors, self.variance, labels
            )

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction="sum")

        # Confidence Loss (Cross Entropy)
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out loss_c from background (conf_t > 0)
        loss_c = loss_c.view(batch_size, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        # take atleast one neg example
        num_neg = torch.clamp(self.negpos_ratio * num_pos, min=1, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # mask_idx has shape (pos + neg examples, 91)
        mask_idx = (pos_idx + neg_idx).gt(0)
        conf_p = conf_data[mask_idx].view(-1, self.num_classes)
        mask = (pos + neg).gt(0)
        conf_gt = conf_t[mask]  # mask has shape (pos + neg examples)
        loss_c = F.cross_entropy(conf_p, conf_gt, reduction="sum")

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N

        return {"loss_l": loss_l, "loss_c": loss_c}
