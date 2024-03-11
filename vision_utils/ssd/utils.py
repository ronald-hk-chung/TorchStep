import torch


def match(threshold, truths, priors, variances, labels):
    """Match each prior box with the ground truth box of the highest jacard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds

    Args:
        thresold (float):   IOU thresold used when matching boxes
        truths (tensor):    Ground truth boxes of shape in XYXY
                            Shape: (num_obj, num_priors)
        priors (tensor):    Prior boxes from priorbox layers in CXCYWH
                            Shape: (n_prior, 4)
        variances (tensor): Variances corresponding to each prior coord
                            Shape: (num_priors, 4)
        labels (tensor):    All class labels for the image
                            Shape: (num_obj, )

    Return:
        tuple(loc_t, conf_t) of matched indices corresponding to
        loc (tensor):       encoded offsets to learn
                            Shape: (num_priors, 4)
        conf (tensor):      top class label for each prior
                            Shape: (num_priors, )
    """
    # Calculate jaccard overalap
    overlaps = jaccard(truths, point_form(priors))
    # if no object present, return all labels background
    if overlaps.shape[0] == 0:
        return 0, 0
    # best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(dim=1)
    # best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(dim=0)
    # make sure every ground truth box has a least one default box that passed the threshold
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # ensure every box matches with its prior of max overlap
    for idx in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[idx]] = idx
    # Get coordinates of ground truth for each prior in XYXY
    matches = truths[best_truth_idx]  # Shape: (num_priors, 4)
    conf = labels[best_truth_idx]  # Shape: (num_priors)
    # label as background best_obj_overlap < thresohold
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)  # matches in XYXY, priors in CXCYWH
    return loc, conf


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes

    Args:
        matched (tensor):           Coordinates of ground truth for each prior in XYXY
                                    Shape: (num_priors, 4)
        priors (tensor):            Prior boxes from priorbox layers in CXCYWH
                                    Shape: (n_prior, 4)
        variances (list[float]):    Variances of priorboxes

    Return:
        encoded_boxes (tensor):     variances encoded matched boxes in CXCYWH
                                    Shape: (num_priors, 4)
    """
    # Distance between match center and prior center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= variances[0] * priors[:, 2:]
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


def decode(loc, priors, variances):
    """Decode locates from predictions using priors to undo encoding at train time

    Args:
        loc (tensor):               location predictions for loc layers,
                                    Shape: (num_priors, 4)
        priors (tensor):            Prior boxes in CXCYWH
                                    Shape: (num_priors, 4)
        variances (list[float]):    Variances of priorboxes

    Return:
        decoded_bboxes (tensor):    decoded bounding boxes in XYXY
                                    Shape: (num_priors, 4)
    """
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes
    The jaccard overlap is simply the intersection over union(IOU) of two boxes
    A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)

    Args:
        box_a (tensor):             Shape: (num_objects, 4)
        box_b (tensor):             Shape: (num_objects, 4)

    Return:
        jaccard_overlap (tensor):   Shape: (box_a.size(0), box_b.size(0))
    """
    inter = intersect(box_a, box_b)
    # A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)

    area_a = (
        ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
        .unsqueeze(1)
        .expand_as(inter)
    )  # [A,B]
    area_b = (
        ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))
        .unsqueeze(0)
        .expand_as(inter)
    )  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def intersect(box_a, box_b):
    """Compute intersection area - A ∩ B"""
    # Get number of coordinates for box_a and box_b
    A = box_a.size(0)
    B = box_b.size(0)
    # resize box to [A, B, 2] and get the minimum of xmax, ymax for box_a and box_b
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        box_b[:, 2:].unsqueeze(0).expand(A, B, 2),
    )
    # resize box to [A, B, 2] and get the maximum of xmin, ymin for box_a and box_b
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A, B, 2),
        box_b[:, :2].unsqueeze(0).expand(A, B, 2),
    )
    # get the intersection coordinates
    inter = torch.clamp((max_xy - min_xy), min=0)
    # return size of intersection
    return inter[:, :, 0] * inter[:, :, 1]


def point_form(boxes):
    """Convert prior_boxes from (CXCYWH) to (XYXY)
    Args:
        boxes (tensor): center-size default boxes from priorbox layers
    Return:
        boxes (tensor): converted xmin, ymin, xmax, ymax form of boxes
    """
    return torch.cat(
        (
            boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
            boxes[:, :2] + boxes[:, 2:] / 2,  # xmax, ymax
        ),
        1,
    )


def log_sum_exp(x):
    """Utility function for computing log_sum_exp log(sum_p(exp(c^p)))
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (tensor): conf_preds from conf layers
    """
    x_max, x_max_indices = torch.max(x, dim=1, keepdim=True)
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max
