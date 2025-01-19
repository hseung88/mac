import torchvision.transforms as transforms
from datasets.datamodule import DataModule


class ImageDataModule(DataModule):
    def __init__(self, config, IMG_MEAN, IMG_STD):
        super().__init__(config)

        self.normalize = transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        self. default_transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        if self.hparams.aug_mult > 0:
            self.hparams.augment = False
            self.aug_transform = self.augmentation_transform()

        self.tr_transform = self.train_transform()
        self.te_transform = self.default_transform

    def augmentation_transform(self):
        if self.hparams.aug_mult > 0:
            raise NotImplementedError("Subclass should implement this method.")

        return None

    def train_transform(self):
        if self.hparams.augment:
            return transforms.Compose([
                transforms.RandomCrop(self.config.dataset.image_size,
                                      padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            return self.default_transform
