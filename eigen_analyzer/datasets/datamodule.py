import inspect
from argparse import Namespace
from torch.utils.data import DataLoader
from common.path import abs_path
from .hooks import DataHooks


class DataModule(DataHooks):
    def __init__(self, config):
        super().__init__()

        self.trainer = None
        self.config = config
        hparams = {
            'data_dir': config.dataset.data_dir,
            'train_batch_size': config.data.train_batch_size,
            'test_batch_size': config.data.test_batch_size,
            'num_workers': config.data.num_workers,
            'pin_memory': config.data.get('pin_memory', True),
            'augment': config.data.get('augment', False),
            'drop_last': config.data.get('drop_last', True),
            'aug_mult': config.data.get('aug_mult', 1)
        }
        self.hparams = Namespace(**hparams)
        self.tr_dataset = None
        self.te_dataset = None
        self.ev_dataset = None

        # is working directory changed?
        return_dir = config.get('return_dir', None)
        if return_dir is not None:
            self.hparams.data_dir = abs_path(return_dir, self.hparams.data_dir)

    @property
    def augmentation_multiplicity(self):
        return self.hparams.aug_mult

    def train_loader(self, batch_size=None):
        assert self.tr_dataset

        batch_size = batch_size or self.hparams.train_batch_size

        return DataLoader(
            dataset=self.tr_dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            shuffle=True
        )

    def test_loader(self, batch_size=None):
        assert self.te_dataset

        batch_size = batch_size or self.hparams.test_batch_size

        return DataLoader(
            dataset=self.te_dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )

    def eval_loader(self, batch_size=None):
        assert self.ev_dataset
        batch_size = batch_size or self.hparams.test_batch_size

        return DataLoader(
            dataset=self.ev_dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )

    @classmethod
    def from_dataset(
            cls,
            train_set, test_set, val_set=None,
            batch_size=1, num_workers=0,
            **datamodule_kwargs
    ):
        def dataloader(dataset, shuffle=False):
            return DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              pin_memory=True)

        def train_dataloader():
            return dataloader(train_set, shuffle=True)

        def test_dataloader():
            return dataloader(test_set)

        def val_dataloader():
            return dataloader(val_set)

        candidate_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
        accepted_params = inspect.signature(cls.__init__).parameters
        accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in accepted_params.values())
        if accepts_kwargs:
            special_kwargs = candidate_kwargs
        else:
            accepted_param_names = set(accepted_params)
            accepted_param_names.discard("self")
            special_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted_param_names}

        datamodule = cls(**datamodule_kwargs, **special_kwargs)
        if train_set is not None:
            datamodule.train_dataloader = train_dataloader
        if test_set is not None:
            datamodule.test_dataloader = test_dataloader
        if val_set is not None:
            datamodule.val_dataloader = val_dataloader

        return datamodule
