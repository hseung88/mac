from utils.exceptions import MisconfigurationException


class DataHooks:
    def __init__(self):
        super().__init__()

    def setup(self, stage: str) -> None:
        """Called at the beginning of fit (train + validate), validate, test, or predict. This is a good hook when
        you need to build models dynamically or adjust something about them. This hook is called on every process
        when using DDP.
        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """

    def train_dataloader(self):
        """Implement one or more PyTorch DataLoaders for training.
        Return:
            A collection of :class:`torch.utils.data.DataLoader` specifying training samples.

        Example::
            # single dataloader
            def train_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=True, transform=transform,
                                download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=True
                )
                return loader
            # multiple dataloaders, return as list
            def train_dataloader(self):
                mnist = MNIST(...)
                cifar = CIFAR(...)
                mnist_loader = torch.utils.data.DataLoader(
                    dataset=mnist, batch_size=self.batch_size, shuffle=True
                )
                cifar_loader = torch.utils.data.DataLoader(
                    dataset=cifar, batch_size=self.batch_size, shuffle=True
                )
                # each batch will be a list of tensors: [batch_mnist, batch_cifar]
                return [mnist_loader, cifar_loader]
            # multiple dataloader, return as dict
            def train_dataloader(self):
                mnist = MNIST(...)
                cifar = CIFAR(...)
                mnist_loader = torch.utils.data.DataLoader(
                    dataset=mnist, batch_size=self.batch_size, shuffle=True
                )
                cifar_loader = torch.utils.data.DataLoader(
                    dataset=cifar, batch_size=self.batch_size, shuffle=True
                )
                # each batch will be a dict of tensors: {'mnist': batch_mnist, 'cifar': batch_cifar}
                return {'mnist': mnist_loader, 'cifar': cifar_loader}
        """
        raise MisconfigurationException("`train_dataloader` must be implemented to be used with the Lightning Trainer")

    def test_dataloader(self):
        r"""
        Implement one or multiple PyTorch DataLoaders for testing.
        For data processing use the following pattern:
            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`
        However, the above are only necessary for distributed processing.
        .. warning:: do not assign state in prepare_data
        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.test`
        - :meth:`prepare_data`
        - :meth:`setup`
        Note:
            If you don't need a test dataset and a :meth:`test_step`, you don't need to implement
            this method.
        Note:
            In the case where you return multiple test dataloaders, the :meth:`test_step`
            will have an argument ``dataloader_idx`` which matches the order here.
        """
        raise MisconfigurationException("`test_dataloader` must be implemented to be used with the Lightning Trainer")

    def val_dataloader(self):
        r"""
        Implement one or multiple PyTorch DataLoaders for validation.
        The dataloader you return will not be reloaded unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_n_epochs` to
        a positive integer.
        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.
        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`
        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.validate`
        - :meth:`prepare_data`
        - :meth:`setup`
        """
        raise MisconfigurationException("`val_dataloader` must be implemented to be used with the Lightning Trainer")
