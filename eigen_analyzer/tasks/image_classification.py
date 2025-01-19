from hydra.utils import instantiate
import torch
import torchvision.transforms as transforms
from torchmetrics import MeanMetric, Accuracy
from utils import _batch_to_device
from .base import MLTask


class ImageClassification(MLTask):
    def __init__(self, config, device):
        super().__init__(config, device)

        self.net = instantiate(config.network.module,
                               **config.network.params)
        self.criterion = instantiate(config.task.loss_func)
        self._optimizer = instantiate(config.optimizer,
                                      self.net.parameters())

        if self.config.get('lr_scheduler'):
            self._lr_scheduler = instantiate(self.config.lr_scheduler, self._optimizer)

        self.num_classes = config.dataset.num_classes

        self.te_loss = MeanMetric()
        self.te_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        self.grad_accum_steps = 1

    def _model_step(self, batch):
        inputs, targets = batch
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)

        return outputs, loss

    def training_step(self, epoch, batch_idx, batch, do_step=True):
        _, loss = self._model_step(batch)

        loss.backward()
        if do_step:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'batch_loss': loss.item()
        }

    def training_step_augmented(self, epoch, batch_idx, batch, do_step=True):
        images, targets = batch
        batch_size = targets.size(0)
        assert self.trainer is not None, "Trainer was not attached to the model."
        K = self.trainer.aug_mult
        transform = self.trainer.data_module.aug_transform

        if K > 0:
            images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
            targets = torch.repeat_interleave(targets, repeats=K, dim=0)
            images = transforms.Lambda(
                lambda x: torch.stack([transform(x_) for x_ in x]))(images_duplicates)
            assert len(images) == K * batch_size

        outputs = self.net(images)
        loss = self.criterion(outputs, targets)

        loss.backward()
        # check if we are at the end of a true batch
        # is_updated = not (self.optimizer._check_skip_next_step(pop_next=False))

        per_param_norms = [
            g.grad_sample.view(len(g.grad_sample), -1).norm(2, dim=-1)
            for g in self.parameters() if g.grad_sample is not None
        ]
        per_sample_norms = (torch.stack(per_param_norms, dim=1).norm(2, dim=1).cpu().tolist())

        self.optimizer.step()

        metrics = {
            'batch_loss': [loss.item()],
            'grad_sample_norms': per_sample_norms[:batch_size],
        }

        return metrics

    def test_step(self, batch):
        _, targets = batch

        outputs, loss = self._model_step(batch)
        batch_loss = self.te_loss(loss)
        batch_acc = self.te_acc(outputs, targets)

        return {'te_loss': batch_loss.item(),
                'te_acc': batch_acc.item()}

    @torch.no_grad()
    def evaluate(self, metrics, train_loader=None, test_loader=None, prefix=None):
        prefix = f'{prefix}_' if prefix else ''

        if train_loader:
            metrics.update(self._evaluate(train_loader, prefix=f'{prefix}tr'))

        if test_loader:
            metrics.update(self._evaluate(test_loader, prefix=f'{prefix}te'))

    @torch.no_grad()
    def _evaluate(self, data_loader, prefix=None):
        self.net.eval()
        if prefix:
            prefix += '_'

        self.te_loss.reset()
        self.te_acc.reset()

        for batch_idx, batch in enumerate(data_loader):
            batch = _batch_to_device(batch, self.device)

            self.test_step(batch)

        metrics = {
            f"{prefix}loss": self.te_loss.compute().item(),
            f"{prefix}acc": self.te_acc.compute().item()
        }
        return metrics

    def forward(self, x):
        self.net(x)
