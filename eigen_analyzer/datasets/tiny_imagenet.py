import os
import torchvision.transforms as transforms
from datasets.image_data import ImageDataModule
from torchvision.datasets import ImageFolder 

class TinyImageNetDataModule(ImageDataModule):
    def __init__(self, config):
        # Use imagenet values
        TINY_IMAGENET_MEAN = (0.480, 0.448, 0.397)  
        TINY_IMAGENET_STD = (0.276, 0.269, 0.282)  
        super().__init__(config, TINY_IMAGENET_MEAN, TINY_IMAGENET_STD)

    @property
    def num_classes(self) -> int:
        return 200  # Tiny ImageNet has 200 classes

    @property
    def shape(self):
        return (3, 64, 64)  # Tiny ImageNet images are 64x64 and RGB

    def augmentation_transform(self):
        return transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def setup(self):
        data_path = '/home/hseung/work/data/tiny-imagenet-200'
        self.tr_dataset = ImageFolder(
            os.path.join(data_path, 'train'), 
            transform=self.tr_transform
        )
        self.te_dataset = ImageFolder(
            os.path.join(data_path, 'val'), 
            transform=self.te_transform
        )
        # Using validation set for evaluation as Tiny ImageNet does not have a separate test set
        self.ev_dataset = ImageFolder(
            os.path.join(data_path, 'val'), 
            transform=self.te_transform
        )
