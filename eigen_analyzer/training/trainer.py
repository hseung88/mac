import time
import sys
import torch
from common.logging import logger as log
from common.path import path_join
from tasks.base import MLTask
from datasets.datamodule import DataModule
from utils import _batch_to_device, print_metrics, CSVLogger


class Trainer:
    """
    Basic non-private trainer

    on_fit_start():
    - move the model to GPU
    - call model.on_fit_start()
    - store the randomly intialized weights if `save_init` is set True

    on_fit_end():
    - flush the logs to files
    - save the last iterate of model parameters if `save_last` is set True
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self._loggers = []
        self.logger = self.add_logger() if config.save else None  # default logger
        self.max_epochs = config.epochs

        # model save
        self.save_init = config.trainer.save_init and config.save
        self.save_best = config.trainer.save_best and config.save
        self.save_last = config.trainer.save_last and config.save
        self.save_every_n_epoch = config.trainer.get('save_every_n_epoch', sys.maxsize)
        self.save_every_n_iters = config.trainer.get('save_every_n_iters', sys.maxsize)

        if self.save_best:
            self.best_perf = 0.0

        self.reset_counters()

    @property
    def epoch(self):
        return self._epoch

    @property
    def steps(self):
        return self._steps

    def reset_counters(self):
        # counters
        self._epoch = 0
        self._steps = 0  # number of updates, i.e., # of calls to optimizer.step()
        self._iters = 0  # number of iteration, i.e., # of processed minibatches

    def add_logger(self, name=None):
        if not self.config.save:
            log.debug('Attempted to add a logger when `save` is set to False.')
            return None

        name = path_join(self.config.working_dir, name) if name else self.config.working_dir

        logger = CSVLogger(name)
        self._loggers.append(logger)
        return logger

    def log_metrics(self, logger, metrics, add_counter=True):
        """
        log metrics to a logger

        add_counter: bool, if True, epoch number and step counter are added to the metrics dictionary
        """
        if logger is None:
            return

        if add_counter:
            metrics['epoch'] = self._epoch
            metrics['step'] = self._steps

        logger.log_metrics(metrics)

    def model_checkpoint(self, model, metrics):
        if self.save_best:
            curr_perf = metrics[model.perf_metric_name]
            if curr_perf > self.best_perf:
                log.debug(f"New best: prev={self.best_perf} curr={curr_perf}")
                self.save_model(model, is_best=True)
                self.best_perf = curr_perf

        if (self._epoch % self.save_every_n_epoch) == 0:
            already_saved = ((self._steps - 1) % self.save_every_n_iters) == 0

            if not already_saved:
                self.save_model(model)

    def on_fit_start(self, model, data_module):
        """
        Move model to the device and call model.on_fit_start()
        """
        self.reset_counters()
        self.data_module = data_module

        model = model.to(self.device)
        model.on_fit_start(self)

        # save the initialization
        if self.save_init:
            self.save_model(model)

        return model

    def on_fit_end(self, model):
        for logger in self._loggers:
            logger.save()

        if self.save_last:
            self.save_model(model, is_best=False)

        self.data_module = None
        model.on_fit_end()

    def fit(self, model: MLTask, data_module: DataModule):
        train_loader = data_module.train_loader()
        test_loader = data_module.test_loader()

        # attach the model to the trainer
        model = self.on_fit_start(model, data_module)

        for self._epoch in range(1, self.max_epochs+1):
            t_epoch_start = time.time()
            epoch_loss = 0.0
            model.train()

            for batch_idx, batch in enumerate(train_loader):
                self._iters += 1
                batch = _batch_to_device(batch, self.device)
                model.training_step(self._epoch, batch_idx, batch)
                self._steps += 1

                # Uncomment the following line if need to save the model at every iteration
                if (self._steps % self.save_every_n_iters) == 0:
                    # iter_metrics = {'epoch': self._epoch,
                    #                 'step': self._steps,
                    #                 'tr_loss': batch_result['batch_loss'],
                    #                 'tr_acc': 0.0}
                    # model.evaluate(iter_metrics,
                    #                train_loader=None, test_loader=test_loader)
                    # iter_metrics['time'] = time.time() - t_epoch_start
                    # print_metrics(iter_metrics, show_header=(self._steps == 0))
                    self.save_model(model)

            # end of an epoch
            if model.lr_scheduler:
                model.lr_scheduler.step()

            epoch_time = time.time() - t_epoch_start
            epoch_loss /= (batch_idx + 1)

            # evaluate the model
            epoch_metrics = {'epoch': self._epoch, 'step': self._steps}
            model.evaluate(epoch_metrics,
                           train_loader=train_loader,
                           test_loader=test_loader)
            epoch_metrics['time'] = epoch_time

            self.log_metrics(self.logger, epoch_metrics, add_counter=False)
            print_metrics(epoch_metrics, show_header=(self._epoch == 1))

            # checkpointing
            self.model_checkpoint(model, epoch_metrics)

        # fitting finished
        self.on_fit_end(model)

    def save_model(self, model, is_best=False):
        if not self.config.save:
            return

        state = {
            'epoch': self.epoch,
            'step': self.steps,
            'model_state': model.state_dict(),
            'optimizer_state': model.optimizer.state_dict(),
        }
        filename = 'best_model' if is_best else f'checkpoint_E{self.epoch}S{self.steps}'

        torch.save(state, f'./checkpoints/{filename}.pt')

    def load_model(self, path, model, device):
        """
        path: path to the checkpoint file (.pt)
        """
        state = torch.load(path, map_location=device)
        self._epoch = state['epoch']
        self._steps = state['step']

        model.load_state_dict(state['model_state'])
