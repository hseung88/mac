# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import os
from datetime import datetime
from argparse import Namespace
from typing import Any, Dict, List, Optional, Set, Union

from torch import Tensor

from lightning.fabric.loggers.logger import Logger, rank_zero_experiment
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.logger import _add_prefix
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.types import _PATH


class CSVLogger(Logger):
    r"""Log to the local file system in CSV format.

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

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        root_dir: _PATH,
        prefix: str = "",
        flush_logs_every_n_steps: int = 10,
    ):
        super().__init__()
        root_dir = os.fspath(root_dir)
        self._root_dir = root_dir
        self._prefix = prefix
        self._fs = get_filesystem(root_dir)
        self._experiment: Optional[_ExperimentWriter] = None
        self._flush_logs_every_n_steps = flush_logs_every_n_steps

    @property
    def name(self) -> str:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment.

        """
        return ""

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
        return self._root_dir

    @property
    @rank_zero_experiment
    def experiment(self) -> "_ExperimentWriter":
        """Actual ExperimentWriter object. To use ExperimentWriter features anywhere in your code, do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        os.makedirs(self._root_dir, exist_ok=True)
        self._experiment = _ExperimentWriter(log_dir=self.log_dir)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:  # type: ignore[override]
        raise NotImplementedError("The `CSVLogger` does not yet support logging hyperparameters.")

    @rank_zero_only
    def log_metrics(  # type: ignore[override]
        self, metrics: Dict[str, Union[Tensor, float]]
    ) -> None:
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        step = len(self.experiment.metrics)
        self.experiment.log_metrics(metrics)
        if (step + 1) % self._flush_logs_every_n_steps == 0:
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

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    NAME_METRICS_FILE = f'metrics_{timestamp}.csv'

    def __init__(self, log_dir: str) -> None:
        self.metrics: List[Dict[str, float]] = []
        self.metrics_keys: List[str] = []

        self._fs = get_filesystem(log_dir)
        self.log_dir = log_dir
        self._fs.makedirs(self.log_dir, exist_ok=True)

        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)

    def log_metrics(self, metrics_dict: Dict[str, float]) -> None:
        """Record metrics."""

        def _handle_value(value: Union[Tensor, Any]) -> Any:
            if isinstance(value, Tensor):
                return value.item()
            return value

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        self.metrics.append(metrics)

    def save(self) -> None:
        """Save recorded metrics into files."""
        if not self.metrics:
            return

        new_keys = self._record_new_keys()
        file_exists = self._fs.isfile(self.metrics_file_path)

        if new_keys and file_exists:
            # we need to re-write the file if the keys (header) change
            self._rewrite_with_new_header(self.metrics_keys)

        with self._fs.open(self.metrics_file_path, mode=("a" if file_exists else "w"), newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics_keys)
            if not file_exists:
                # only write the header if we're writing a fresh file
                writer.writeheader()
            writer.writerows(self.metrics)

        self.metrics = []  # reset

    def _record_new_keys(self) -> Set[str]:
        """Records new keys that have not been logged before."""
        current_keys = set().union(*self.metrics)
        new_keys = current_keys - set(self.metrics_keys)
        self.metrics_keys.extend(new_keys)
        return new_keys

    def _rewrite_with_new_header(self, fieldnames: List[str]) -> None:
        with self._fs.open(self.metrics_file_path, "r", newline="") as file:
            metrics = list(csv.DictReader(file))

        with self._fs.open(self.metrics_file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)
