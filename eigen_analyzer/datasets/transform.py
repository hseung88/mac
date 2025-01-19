import torch
import torchvision.transforms as transforms
from common.logging import logger as log


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    elif config.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.rescaled:
        X = 2 * X - 1.0
    elif config.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)


def batch_augmentation(aug_fn, aug_mult):
    """
    Batch augmentation

    Parameters:
    -------------------
    aug_fn: a set of transformations to apply
    aug_mult: int, augmentation multiplicity
    """
    if aug_mult < 1:
        log.error(f"Invalid augmentation multiplicity value: {aug_mult}")

    return transforms.Lambda(
        lambda x: torch.stack([aug_fn(x) for _ in range(aug_mult)])
    )
