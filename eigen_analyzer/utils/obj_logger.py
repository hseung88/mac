import os
from argparse import Namespace
from typing import Any, Dict, Optional, Union
import dill as pickle
from torch import Tensor
from lightning.fabric.loggers.logger import Logger, rank_zero_experiment
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.types import _PATH
from common.path import path_join


class OBJLogger(Logger):
    r"""Log to the local file system in dill format.

    Logs are saved to ``os.path.join(root_dir, name, version)``.

    Args:
        root_dir: The root directory in which all your experiments with different names and versions will be stored.
        name: Experiment name. Defaults to ``'lightning_logs'``.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        prefix: A string to put at the beginning of metric keys.
        flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).

    Example::

        from lightning.fabric.loggers import CSVLogger

        logger = CSVLogger("path/to/logs/root", name="my_model")
        logger.log_metrics({"loss": 0.235, "acc": 0.75})
        logger.finalize("success")

    """
    def __init__(
        self,
        root_dir: _PATH,
        name: str,
        flush_logs_every_n_steps: int = 5,
    ):
        super().__init__()
        root_dir = os.fspath(root_dir)
        self._root_dir = root_dir
        self._name = name
        self._log_dir = path_join(root_dir, name)
        self._fs = get_filesystem(root_dir)
        self._experiment: Optional[_ExperimentWriter] = None
        self._flush_logs_every_n_steps = flush_logs_every_n_steps

    @property
    def name(self) -> str:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment.

        """
        return self._name

    @property
    def version(self) -> Union[int, str]:
        """Gets the version of the experiment.

        Returns:
            The version of the experiment if it is specified, else the next version.

        """
        return ""

    @property
    def root_dir(self) -> str:
        """Gets the save directory where the versioned CSV experiments are saved."""
        return self._root_dir

    @property
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        # create a pseudo standard path
        return self._log_dir

    @property
    @rank_zero_experiment
    def experiment(self) -> "_ExperimentWriter":
        """Actual ExperimentWriter object. To use ExperimentWriter features anywhere in your code, do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        os.makedirs(self._log_dir, exist_ok=True)
        self._experiment = _ExperimentWriter(log_dir=self.log_dir)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:  # type: ignore[override]
        raise NotImplementedError("The `OBJLogger` does not yet support logging hyperparameters.")

    @rank_zero_only
    def log_metrics(  # type: ignore[override]
        self, metrics: Dict[str, Union[Tensor, float]]
    ) -> None:
        step = len(self.experiment.metrics)
        if (step + 1) % self._flush_logs_every_n_steps == 0:
            self.save()

    @rank_zero_only
    def log_object(self, epoch, step, metrics) -> None:
        exp_len = len(self.experiment.metrics)
        self.experiment.log_object(epoch, step, metrics)
        if (exp_len + 1) % self._flush_logs_every_n_steps == 0:
            self.save()

    @rank_zero_only
    def save(self) -> None:
        super().save()
        self.experiment.save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._experiment is None:
            # When using multiprocessing, finalize() should be a no-op on the main process, as no experiment has been
            # initialized there
            return
        self.save()


class _ExperimentWriter:
    r"""Experiment writer for CSVLogger.

    Args:
        log_dir: Directory for the experiment logs

    """

    def __init__(self, log_dir: str) -> None:
        self.metrics = []
        self._fs = get_filesystem(log_dir)
        self.log_dir = log_dir
        self._fs.makedirs(self.log_dir, exist_ok=True)

    def log_object(self, epoch, step, metric_obj) -> None:
        """Record metrics."""
        self.metrics.append((epoch, step, metric_obj))

    def save(self) -> None:
        """Save recorded metrics into files."""
        if not self.metrics:
            return

        for epoch, step, metric_obj in self.metrics:
            with open(path_join(self.log_dir, f'E{epoch}S{step}.dill'), 'wb') as fout:
                pickle.dump(metric_obj, fout)

        self.metrics.clear()  # reset
