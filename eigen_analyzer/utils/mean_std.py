import argparse
import torch
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


data_dir = "../../../Experiment/Dataset/pytorch_data"
data_class = {
    'mnist': MNIST,
    'fmnist': FashionMNIST,
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
}


def main(args):
    dataset = data_class[args.dname](data_dir, train=True, download=False,
                                     transform=transforms.Compose([transforms.ToTensor()]))

    loader = DataLoader(dataset, batch_size=512, num_workers=args.num_workers, shuffle=False)
    data_len = len(dataset)

    mean, sq_mean, avg_std = 0., 0., 0.
    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)

        mean += images.mean(2).sum(0)
        sq_mean += (images ** 2).mean(2).sum(0)
        avg_std += images.std(2).sum(0)    # some papers compute the std in this manner

    mean /= data_len
    avg_std /= data_len
    sq_mean /= data_len
    std = torch.sqrt(sq_mean - (mean ** 2))
    mean = mean.numpy()
    std = std.numpy()
    avg_std = avg_std.numpy()
    print(f"Mean: {mean}")
    print(f"STD : {std}")
    print(f"Reference STD: {avg_std}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dname', type=str, default='cifar10')
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    main(args)
