from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from datasets.image_data import ImageDataModule


class CIFAR100DataModule(ImageDataModule):
    def __init__(self, config):
        # normalize = transforms.Normalize(
        #     mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #     std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        # )
        CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
        CIFAR100_STD = (0.2673, 0.2564, 0.2762)

        super().__init__(config, CIFAR100_MEAN, CIFAR100_STD)

    @property
    def num_classes(self) -> int:
        return 100

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
        self.tr_dataset = CIFAR100(
            self.hparams.data_dir,
            train=True,
            download=False,
            transform=self.tr_transform
        )
        self.te_dataset = CIFAR100(
            self.hparams.data_dir,
            train=False,
            download=False,
            transform=self.te_transform
        )
        # to evaluate the performance on the training dataset
        self.ev_dataset = CIFAR100(
            self.hparams.data_dir,
            train=True,
            download=False,
            transform=self.te_transform
        )
