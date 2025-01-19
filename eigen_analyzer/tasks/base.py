import torch.nn as nn


class MLTask(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.device = device
        self._trainer = None
        self._optimizer = None
        self._lr_scheduler = None
        self.perf_metric_name = 'te_acc'

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, value):
        self._trainer = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    def configure_optimizer(self) -> None:
        """ initialize the optimizer and lr_scheduler """

    def on_fit_start(self, trainer):
        self.trainer = trainer

        if hasattr(self._optimizer, '_model'):
            self._optimizer._configure(trainer, self)

        self._optimizer.zero_grad()

    def on_fit_end(self) -> None:
        """ end of fitting """
        self.trainer = None
