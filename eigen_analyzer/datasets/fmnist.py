from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from datasets.image_data import ImageDataModule

class FashionMNISTDataModule(ImageDataModule):
    def __init__(self, config):
        FASHION_MNIST_MEAN = (0.286040,)  # Mean for Fashion MNIST
        FASHION_MNIST_STD = (0.353024,)  # Std Dev for Fashion MNIST

        super().__init__(config, FASHION_MNIST_MEAN, FASHION_MNIST_STD)

    @property
    def num_classes(self) -> int:
        return 10  # Fashion MNIST has 10 classes

    @property
    def shape(self):
        return (1, 28, 28)  # Fashion MNIST images are 28x28 and grayscale

    def augmentation_transform(self):
        return transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    def setup(self):
        self.tr_dataset = FashionMNIST(
            self.hparams.data_dir,
            train=True,
            download=True,
            transform=self.tr_transform
        )
        self.te_dataset = FashionMNIST(
            self.hparams.data_dir,
            train=False,
            download=True,
            transform=self.te_transform
        )
        # to evaluate the performance on the training dataset
        self.ev_dataset = FashionMNIST(
            self.hparams.data_dir,
            train=True,
            download=True,
            transform=self.te_transform
        )

