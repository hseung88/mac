from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from datasets.image_data import ImageDataModule


# https://github.com/deepmind/jax_privacy/blob/main/jax_privacy/src/training/image_classification/data/mnist_cifar_svhn.py
# means = jnp.array([0.49139968, 0.48215841, 0.44653091])
#  stds = jnp.array([0.24703223, 0.24348513, 0.26158784])


class CIFAR10DataModule(ImageDataModule):
    def __init__(self, config):
        # normalize = transforms.Normalize(
        #     mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #     std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        # )
        CIFAR10_MEAN = (0.491396, 0.482158, 0.446530)
        CIFAR10_STD = (0.247032, 0.243484, 0.2616158)

        super().__init__(config, CIFAR10_MEAN, CIFAR10_STD)

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def shape(self):
        return (3, 32, 32)

    def augmentation_transform(self):
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    def setup(self):
        self.tr_dataset = CIFAR10(
            self.hparams.data_dir,
            train=True,
            download=False,
            transform=self.tr_transform
        )
        self.te_dataset = CIFAR10(
            self.hparams.data_dir,
            train=False,
            download=False,
            transform=self.te_transform
        )
        # to evaluate the performance on the training dataset
        self.ev_dataset = CIFAR10(
            self.hparams.data_dir,
            train=True,
            download=False,
            transform=self.te_transform
        )
