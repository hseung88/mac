from .cifar import CIFAR10DataModule
from .cifar100 import CIFAR100DataModule
from .fmnist import FashionMNISTDataModule
from .tiny_imagenet import TinyImageNetDataModule

__all__ = ['CIFAR10DataModule', 'CIFAR100DataModule', 'FashionMNISTDataModule', 'TinyImageNetDataModule']