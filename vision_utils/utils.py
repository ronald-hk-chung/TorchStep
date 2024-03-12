from torchvision import transforms
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Callable
import numpy as np


def show_batch(
    dataloader: DataLoader,
    labelling: Callable,
    transforms: transforms.Compose = T.ToPILImage(),
):
    """
    Function to show a batch of images with custom labelling

    Args:
        dataloader (DataLoader): DataLoader of images to show
        transforms (transforms): transform of image tensor to PIL
        labelling (Callable): function to return label given labels generated from dataloader
    """
    fig = plt.figure(figsize=(20, 10))
    imgs, *labels = next(iter(dataloader))
    nrows, ncolumns = 4, 8
    for i, label in enumerate(list(zip(*labels))):
        plt.subplot(nrows, ncolumns, i + 1)
        plt.imshow(transforms(imgs[i]))
        plt.title(labelling(label))
        plt.axis(False)


def show_image(
    img: np.ndarray,
    bbs: list = None,
    bbs_format: str = "XYXY",
    bbs_label: list = None,
    color: str = "r",
):
    """
    Show Image

    Args:
        img (ndarray): image to show
        bbs (list): list bounding boxes
        bbs_format (str): bounding box format ('XYXY', 'XYWH', 'CXCYWH'), default to 'XYXY'
        bbs_label (list): list of bounding box label
        color (str): edgecolor of bounding box
    """
    fig, ax = plt.subplots()
    ax.imshow(img)
    if bbs is not None:
        for i, bb in enumerate(bbs):
            if bbs_format == "XYXY":
                x1, y1, x2, y2 = bb
                w, h = x2 - x1, y2 - y1
            elif bbs_format == "XYWH":
                x1, y1, w, h = bb
            elif bbs_format == "CXCYWH":
                cx, cy, w, h = bb
                x1 = cx - w / 2
                y1 = cy - h / 2
            rect = patches.Rectangle(
                (x1, y1), w, h, linewidth=1, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            if bbs_label is not None:
                ax.annotate(bbs_label[i], (x1, y1), color=color)

    plt.axis(False)
    plt.show()
