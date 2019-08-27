import logging
import os
from collections import ChainMap
import pickle
import numpy as np

from delira.training.callbacks import AbstractCallback


from mscl.utils.metrics import AbstractMetric
from mscl.utils import ConfigHandlerAbstract

logger = logging.getLogger(__name__)

#TODO: remove


class Evaluator(AbstractCallback):
    def __init__(self,
                 ch: ConfigHandlerAbstract,
                 batch_metrics={},
                 dataset_metrics={},
                 eval_keys=None,
                 logger_postfix='',
                 start_epoch=1):
        """
        Caches selected result from current epoch and provides functionality
        to compute and log metrics over batches and complete data sets.
        Accepts inputs in form of nested dicts, e.g.
        'task'
            'pred'
            'gt'
        every key can be specified inside the used metric. Because the entire
        batch dict is passed to the metrics, it is possible to define 
        new individual metrics by subclassing `AbstractMetric`.

        Parameters
        ----------
        ch : ConfigHandlerAbstract
            ConfigHandler to provide necessary settings
            is also forwarded to the metric functions
        batch_metrics : dict
            dict with function/metrics which should be evaluated on a per batch
            basis. All function must be wrapped by AbstractMetric in order
            to support proper logging capabilities
            the key determines the name of the metric
        dataset_metrics : dict
            dict with function/metrics which should be evaluated on the whole
            dataset. All function must be wrapped by AbstractMetric in order
            to support proper logging capabilities
            the key determines the name of the metric
        eval_keys : list
            list with keys to be cached from the batch.
        logger_postfix : string
            postfix is applied to all plot titles of metrics
        """
        super().__init__()

        self.ch = ch
        self.batch_metrics = {}
        self.dataset_metrics = {}

        for key, fct in batch_metrics.items():
            if isinstance(fct, AbstractMetric):
                self.batch_metrics[key] = fct
            else:
                logger.warning(
                    f"{key} needs to be a subclass of AbstractMetric")

        for key, fct in dataset_metrics.items():
            if isinstance(fct, AbstractMetric):
                self.dataset_metrics[key] = fct
            else:
                logger.warning(
                    f"{key} needs to be a subclass of AbstractMetric")

        self.dataset_epoch_data = []
        self.dataset_save_keys = eval_keys

        self.logger_postfix = logger_postfix

        self.epoch = start_epoch

    def __call__(self, results):
        """
        Save keys from results to ``dataset_epoch_data``

        Parameters
        ----------
        results : dict
            dict with network results
            results must be in appropriate format for metrics
        """
        # save prediction data for dataset functions
        eval_dict = {}
        if self.dataset_save_keys is None:
            self.dataset_save_keys = results.keys()

        for key in self.dataset_save_keys:
            if key in results:
                eval_dict[key] = results[key]
            else:
                raise ValueError(f"prediction_key {key} "
                                 f"was not in prediction dict")
        self.dataset_epoch_data.append(eval_dict)

    def at_epoch_begin(self, trainer, **kwargs) -> dict:
        """
        Do nothing at epoch begin
        """
        return {}

    def at_epoch_end(self, trainer, **kwargs) -> dict:
        """
        Log Dataset Metrics at end of epoch

        Parameters
        ----------
        trainer : AbstractNetworkTrainer
            network trainer (only used to determine fold)
        kwargs :
            additional keyword arguments (not used)

        Returns
        -------
        dict
            empty
        """
        self.log_dataset_metrics(fold=trainer.fold, new_epoch=True)
        return {}

    def comp_batch_metrics(self):
        """
        Computes the batch metrics
        """
        # check if there is any data
        if not self.dataset_epoch_data:
            return

        # evaluate current batch
        for key, fct in self.batch_metrics.items():
            fct(self.dataset_epoch_data[-1], epoch=self.epoch, ch=self.ch)
        return

    def comp_dataset_metrics(self, new_epoch=False):
        """
        Computes the dataset metrics

        Parameters
        ----------
        new_epoch: bool
            start a new epoch -> data in self.results_batch_epoch
            are cleared
        """
        # check if there is any data
        if not self.dataset_epoch_data:
            return

        # evaluate current dataset
        for key, fct in self.dataset_metrics.items():
            fct(self.dataset_epoch_data, epoch=self.epoch, ch=self.ch)

        if new_epoch:
            self.new_epoch()
        return

    def log_batch_metrics(self, fold=0):
        """
        Log and compute batch metrics

        Parameters
        ----------
        fold: int
            fold is forwarded to MetricLogging to determine correct environment
        """
        # check if there is any data
        if not self.dataset_epoch_data:
            return

        self.comp_batch_metrics()
        for key, fct in self.batch_metrics.items():
            fct.log_metric(name=key + self.logger_postfix,
                           epoch=self.epoch,
                           fold=fold)
        return

    def log_dataset_metrics(self, fold=0, new_epoch=False):
        """
        Log and compute dataset metrics

        Parameters
        ----------
        fold: int
          fold is forwarded to MetricLogging to determine correct environment
        new_epoch: bool
          starts a new epoch
        """
        # check if there is any data
        if not self.dataset_epoch_data:
            return

        self.comp_dataset_metrics()
        for key, fct in self.dataset_metrics.items():
            fct.log_metric(name=key + self.logger_postfix,
                           epoch=self.epoch,
                           fold=fold)
        if new_epoch:
            self.new_epoch()
        return

    def new_epoch(self):
        """
        Starts a new epoch
        """
        self.dataset_epoch_data = []
        self.epoch += 1

    def get_scalar_batch_metrics(self, epoch=None):
        """
        Returns dict with all scalar metrics from last batch
        """
        if epoch is None:
            epoch = self.epoch - 1   # epoch counter is always one ahead

        metrics = {}
        for key, val in self.batch_metrics.items():
            if epoch not in val.metric:
                raise ValueError("Epoch {} is not saved in {}".format(epoch,
                                                                      key))
            # check if list is not empty and metric is scalar
            if (val.metric[epoch]) and ('scalar' in val.metric[epoch][0]):
                metrics[key] = \
                    np.mean([i['scalar'] for i in val.metric[epoch]])
        return metrics

    def get_scalar_dataset_metrics(self, epoch=None):
        """
        Returns a dict with all scalar metrics from dataset_metrics
        """
        if epoch is None:
            epoch = self.epoch - 1  # epoch counter is always one ahead

        metrics = {}
        for key, val in self.dataset_metrics.items():
            if epoch not in val.metric:
                raise ValueError(f"Epoch {epoch} is not saved in {key}")
            if 'scalar' in val.metric[epoch]:
                metrics[key] = val.metric[epoch]['scalar']
        return metrics

    def save_evaluator(self, path):
        """
        Pickle current state of evaluator
        """
        save_path = os.path.join(path,
                                 f'evaluator{self.logger_postfix}.pickle')
        with open(save_path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_evaluator(path):
        """
        Load saved state of evaluator
        """
        with open(path, 'rb') as handle:
            evaluator = pickle.load(handle)
        return evaluator


class EvaluatorSave(Evaluator):
    """
    Provides additional functionality so save metric plots
    """

    def __init__(self, save_freq, save_path=None, *args, **kwargs):
        """
        Saves metric plots during training

        Parameters
        ----------
        save_freq: int
            defines the frequency with which metrics are saved
            (defines the epoch!)
        save_path: string
            defines the path where to save plots
            Can also be provided in a later stage
        args:
            forwarded to evaluator
        kwargs:
            forwarded to evaluator
        """
        super().__init__(*args, **kwargs)
        self.save_freq = save_freq

        if save_path is None:
            logger.info("Savepath for Evaluator was not provided "
                        "at initialization time.")
        self._save_path = save_path

    def comp_dataset_metrics(self, *args, **kwargs):
        """
        Same as evaluator.comp_dataset_metrics, but with additional
        saving stage for metrics
        """
        super().comp_dataset_metrics(*args, **kwargs)

        if self.epoch % self.save_freq == 0:
            # create new directory
            p = os.path.join(self._save_path, f"epoch{self.epoch}")
            self._save_metric_plots(p)

    def _save_metric_plots(self, p):
        """
        Save metric plots to path p
        """
        try:
            os.mkdir(p)
        except OSError:
            logger.error("Creation of the directory %s failed" % p)
            raise OSError("Creation of the directory %s failed" % p)

        # find last valid epoch
        failed = True
        while failed:
            # iterate functions and save plots
            try:
                for key, fct in \
                        ChainMap(self.dataset_metrics, self.batch_metrics).items():
                    fct.plot_metric(epoch=self.epoch,
                                    show=False,
                                    save_path=os.path.join(p, f'{key}'))
                failed = False
            except KeyError:
                logger.warning(f'Epoch {self.epoch} not found in metrics, '
                               f'reducing epoch by one!')
                self.epoch -= 1
                # check for infinite loop
                if self.epoch < 0:
                    logger.error(f'Plot metric not saved.')
                    failed = False

    def save_evaluator(self, path=None):
        """
        Saves evaluator and metric plots
        """
        if path is None:
            path = self._save_path
        super().save_evaluator(path=path)
        p = os.path.join(self._save_path, f"end_save")
        self._save_metric_plots(p)

    @property
    def save_path(self):
        """
        Getter for _save_path
        """
        return self.save_path

    @save_path.setter
    def save_path(self, value):
        """
        Setter for _save_path
        """
        # warn if save path was already provided
        if self._save_path is not None:
            logger.warning(f"Save path was already set to {self._save_path}, "
                           f"new save path is now {value}")
        # create directory
        p = os.path.join(value, 'evaluator' + self.logger_postfix)
        try:
            os.mkdir(p)
        except OSError:
            logger.error("Creation of the directory %s failed" % p)
            raise OSError("Creation of the directory %s failed" % p)
        self._save_path = p
