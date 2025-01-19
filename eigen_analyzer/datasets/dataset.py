import torch
from torch.utils.data import Dataset


class AugMultiDataset(Dataset):
    """
    Wrapper class for multiple data augmentation
    """
    def __init__(self, dataset, aug_mult=16):
        super().__init__()

        self._dataset = dataset
        self.aug_mult = aug_mult

    @property
    def transform(self):
        return self._dataset.transform

    @transform.setter
    def transform(self, trans):
        self._dataset.tranform = trans

    def __getitem__(self, i):
        image, target = self._dataset[i]

        aug_imgs = [self._dataset[i][0] for _ in range(self.aug_mult-1)]
        aug_imgs.append(image)

        return torch.stack(aug_imgs), target

    def __len__(self):
        return self._dataset.__len__()
