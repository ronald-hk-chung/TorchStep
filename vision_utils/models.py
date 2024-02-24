import torch
import torchvision
from torchvision.transforms import v2 as T


def get_pretrained_model(name: str, pretrained_weights: str | None = None):
    """Get pretrained model and pretrained transformation (forward and reverse)

    Args:
    model[str]: name of pretrained model
    weights[str]: name of pretrained model weights

    Returns:
    A tuple of (model, forward_transformation, reverse_transformation)

    Example usage:
    model, ftfms, rtfms = get_prerained_model(name='resnet18',
                                              weights='ResNet18_Weights.IMAGENET1K_V1')
    """

    # Change get_state_dict from Torch Hub
    def get_state_dict_from_hub(self, *args, **kwargs):
        kwargs.pop("check_hash")
        return torch.hub.load_state_dict_from_url(self.url, *args, **kwargs)

    torchvision.models._api.WeightsEnum.get_state_dict = get_state_dict_from_hub

    # Get default transformation and re-construct forward transformation and reverse transformation using V2
    if pretrained_weights is not None:
        weights = torchvision.models.get_weight(pretrained_weights)
        pretrained_transforms = weights.transforms()
        forward_transforms = T.Compose(
            [
                T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
                T.Resize(
                    size=pretrained_transforms.resize_size,
                    interpolation=pretrained_transforms.interpolation,
                    antialias=True,
                ),
                T.CenterCrop(size=pretrained_transforms.crop_size),
                T.Normalize(
                    mean=pretrained_transforms.mean, std=pretrained_transforms.std
                ),
            ]
        )
        reverse_transforms = T.Compose(
            [
                T.Normalize(
                    mean=[0.0] * 3,
                    std=list(map(lambda x: 1 / x, pretrained_transforms.std)),
                ),
                T.Normalize(
                    mean=list(map(lambda x: -x, pretrained_transforms.mean)),
                    std=[1.0] * 3,
                ),
                T.ToPILImage(),
            ]
        )
    else:
        weights = None
        forward_transforms = T.Compose(
            [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
        )
        reverse_transforms = T.ToPILImage()

    # Get model using torchvision.models.get_model
    model = torchvision.models.get_model(name=name, weights=weights)

    return model, forward_transforms, reverse_transforms
