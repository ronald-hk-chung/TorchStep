import torch
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from torchvision.transforms import v2 as T
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colormaps
from typing import Callable
import numpy as np


def get_cam(
    img_path: str,
    tfms: transforms.Compose,
    rtfms: transforms.Compose,
    classifier: Callable,
    layers_to_activate: list,
    class_to_activate: int,
):
    """
    Method to get Class Activation Maps(CAM) for in image

    Args:
      img_path (str):   path of img to be analysed
      tfms (transforms): transform of image, PIL to Tensor
      rtfms (transforms): reverse transform of image, Tensor to PIL
      classifier (TSEngine): TSEngine class classifier
      layers_to_activate (list): layers to get activation from model in classifier
      class_to_activate (int): class for Class Activation Maps (CAM) analysis
    """
    gradients = None
    activations = None

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output

    def forward_hook(module, args, output):
        nonlocal activations
        activations = output

    # unfreeze layers and attach hooks to layers_to_activate
    classifier.unfreeze()
    classifier.attach_forward_hooks(layers_to_activate, forward_hook)
    classifier.attach_backward_hooks(layers_to_activate, backward_hook)

    # Open image and perform forward and backward pass on class to activates
    img = tfms(Image.open(img_path).convert("RGB"))
    y_logits = classifier.model(img.unsqueeze(dim=0).to("cuda"))
    print(f"Prediction: {y_logits}")
    y_logits[class_to_activate].backward()

    # freeze classifier and remove hooks
    classifier.remove_hooks()
    classifier.freeze()

    # Calculate gradient heatmap
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    for i in range(activations.size()[1]):
        activations[:, i, ::] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.nn.functional.relu(heatmap)
    heatmap /= torch.max(heatmap)

    # Plot out original image and image with heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(rtfms(img))
    ax1.axis(False)
    overlay = to_pil_image(heatmap.detach(), mode="F").resize(
        (img.shape[1], img.shape[2]), resample=Image.BICUBIC
    )
    cmap = colormaps["jet"]
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    ax2.imshow(rtfms(img))
    ax2.imshow(overlay, alpha=0.4, interpolation="nearest")
    ax2.axis(False)
    plt.show()
