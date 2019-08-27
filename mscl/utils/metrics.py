import logging
from abc import abstractmethod
from collections import OrderedDict
import matplotlib.pyplot as plt
import pickle
from pylab import savefig

logger = logging.getLogger(__name__)

PLT_SAVE_TYPES = ['.pdf', '.png']

#TODO: remove
class AbstractMetric:
    def __init__(self, metric_fct, pre_processing=False, **kwargs):
        """
        Interface for arbitrary metrics with logging capability

        Parameters
        ----------
        metric_fct: function
            function which computes some kind of metric
        pre_processing: bool
            signals Evaluator if the pre_processing function
            is need for this metric
        """
        super().__init__()
        self.metric = OrderedDict()
        self.metric_fct = metric_fct
        self.pre_processing = pre_processing

    @abstractmethod
    def __call__(self, results, epoch, **kwargs):
        """
        Computes the metric with metric_fct
        Parameters
        ----------
        results: dict
            results dict with prediction and gt data
        """
        raise NotImplementedError

    @abstractmethod
    def log_metric(self, name, epoch, fold=1):
        """
        Function to create correct logging call for metric
        """
        raise NotImplementedError

    @abstractmethod
    def plot_metric(self, epoch, show=False, save_path=None, **kwargs):
        """
        Provide functionality to plot with matplotlib and save result
        to save_path
        """
        raise NotImplementedError


class ScalarBatchMetric(AbstractMetric):
    def __init__(self, metric_fct, result_key, pred_key, gt_key, **kwargs):
        """
        Wrapper for simple scalar metrics for batches
        results dict for call function should be structured as follows:
            ``result_key``
                ``pred_key``
                ``gt_key``

        Parameters
        ----------
        metric_fct : callable
            function to compute metric
        result_key : hashable
            key where prediction and ground truth are located
        pred_key : hashable
            key for prediction
        gt_key : hashable
            key for ground truth
        kwargs :
            additional keyword arguments
        """
        super().__init__(metric_fct, **kwargs)
        self.result_key = result_key
        self.pred_key = pred_key
        self.gt_key = gt_key

    def __call__(self, results, epoch, **kwargs):
        """
        Compute metric

        Parameters
        ----------
        results : dict of dict
            nested dictionary of the form:
            ``result_key``
                ``pred_key``
                ``gt_key``
            where ``pred_key`` contains the prediction and ``gt_key``contains
            the ground truth
        epoch : int
            current epoch

        Returns
        -------
        float
            results of scalar metric
        """
        # if new epoch, create empty list
        if epoch not in self.metric:
            self.metric[epoch] = []

        # compute metric
        scalar = self.metric_fct(results[self.result_key][self.pred_key],
                                 results[self.result_key][self.gt_key])

        # save result
        self.metric[epoch].append({'scalar': scalar})
        return scalar

    def log_metric(self, name, epoch, fold=1):
        """
        Log scalar metric

        Parameters
        ----------
        name : str
            name of metric
        epoch : int
            current epoch
        fold : int, optional
            current fold, by default 1
        """
        logger.info(
            {"value": {"value": self.metric[epoch][-1]['scalar'],
                       "name": name,
                       "env_appendix": "_%02d" % fold, }
             })

    def plot_metric(self, epoch, show=False, save_path=None, close=True,
                    **kwargs):
        """
        Plot scalar value over iterations.

        Parameters
        ----------
        epoch : int
            current epoch
        show : bool, optional
            show plot during training, by default False
        save_path : str, optional
            path where to save plot. If None, plot is not saved,
            by default None
        close : bool, optional
            close plot, by default True
        """
        try:
            plt.plot([i['scalar'] for _, b in self.metric.items() for i in b])
            plt.grid(True)
            plt.tight_layout()
            plt.xlabel('#iteration')
            try:
                plt.ylabel(self.metric_fct.__name__)
            except AttributeError:
                logger.warning(f"y-label failed for {self.__class__.__name__}")

            if show:
                plt.show()
            if save_path is not None:
                for ext in PLT_SAVE_TYPES:
                    savefig(save_path + ext)
            plt.close()
        except:
            logger.warning(f'Plotting failed for epoch {epoch} in '
                           f'{self.__class__.__name__}')


class ScalarDatasetMetric(AbstractMetric):
    def __init__(self, metric_fct, results_key=None, **kwargs):
        """
        Wrapper to compute metric over entire dataset.

        Parameters
        ----------
        metric_fct : callable
            function to compute metric
        result_key : hashable
            key where prediction and ground truth are located
        kwargs :
            additional keyword arguments
        """
        super().__init__(metric_fct, **kwargs)
        self.results_key = results_key

    def __call__(self, results, epoch, **kwargs):
        """
        Compute metric

        Parameters
        ----------
        results : dict of dict
            nested dictionary of the form:
            ``result_key``
                ``pred_key``
                ``gt_key``
            where ``pred_key`` contains the prediction and ``gt_key``contains
            the ground truth
        epoch : int
            current epoch

        Returns
        -------
        float
            results of scalar metric
        """
        if self.results_key is None:
            scalar = self.metric_fct(results)
        else:
            scalar = self.metric_fct([i[self.results_key] for i in results])
        self.metric[epoch] = ({'scalar': scalar})
        return scalar

    def log_metric(self, name, epoch, fold=1):
        """
        Log scalar metric

        Parameters
        ----------
        name : str
            name of metric
        epoch : int
            current epoch
        fold : int, optional
            current fold, by default 1
        """
        logger.info(
            {"value": {"value": self.metric[epoch]['scalar'],
                       "name": name,
                       "env_appendix": "_%02d" % fold, }
             })

    def plot_metric(self, epoch, show=False, save_path=None, close=True,
                    **kwargs):
        """
        Plot scalar value over epochs.

        Parameters
        ----------
        epoch : int
            current epoch
        show : bool, optional
            show plot during training, by default False
        save_path : str, optional
            path where to save plot. If None, plot is not saved,
            by default None
        close : bool, optional
            close plot, by default True
        """
        plt.plot([i['scalar'] for _, i in self.metric.items()])
        plt.grid(True)
        plt.tight_layout()
        plt.xlabel('#epoch')
        plt.ylabel(self.metric_fct.__name__)
        if show:
            plt.show()
        if save_path is not None:
            for ext in PLT_SAVE_TYPES:
                savefig(save_path + ext)
        if close:
            plt.close()


class LineDatasetMetric(ScalarDatasetMetric):
    def __init__(self, metric_fct, results_key=None, **kwargs):
        """
        Wrapper for more complex metrics metric_fct gets a list with all
        esult_dicts from last epoch each dict has a structure defined by the network
        metric_fct should return a scalar value and data which should be a dict
        with keys, 'x' and 'y'. x,y values have to have same dimension and need
        to be of dimensions MxN where M is the number of points and N is the
        number of different lines

        Parameters
        ----------
        metric_fct: fct
            function to compute metric from
        results_key: string
            result_key will be used to select data from results dict
            if None, results dict will be forwarded to metric_fct
        """
        super().__init__(metric_fct, results_key, **kwargs)

    def __call__(self, results, epoch, **kwargs):
        """
        Compute metric

        Parameters
        ----------
        results : dict
            list of batch dictionaries
        epoch : int
            current epoch

        Returns
        -------
        any
            result of ``metric_fct``
        """
        if self.results_key is None:
            line = self.metric_fct(results)
        else:
            line = self.metric_fct([i[self.results_key] for i in results])
        self.metric[epoch] = ({'line': line})
        return line

    def log_metric(self, name, epoch, fold=1):
        """
        Log line metric

        Parameters
        ----------
        name : str
            name of metric
        epoch : int
            current epoch
        fold : int, optional
            current fold, by default 1
        """
        logger.info(
            {"lineplot": {"y_vals": self.metric[epoch]['line']['y'],
                          "x_vals": self.metric[epoch]['line']['x'],
                          "name": name,
                          "env_appendix": "_%02d" % fold, }
             })

    def plot_metric(self, epoch, show=False, save_path=None, close=True,
                    **kwargs):
        """
        Plot line metric

        Parameters
        ----------
        epoch : int
            current epoch
        show : bool, optional
            show plot during training, by default False
        save_path : str, optional
            path where to save plot. If None, plot is not saved,
            by default None
        close : bool, optional
            close plot, by default True
        """
        plt.plot(self.metric[epoch]['line']['x'],
                 self.metric[epoch]['line']['y'])
        plt.grid(True)
        plt.tight_layout()
        if show:
            plt.show()
        if save_path is not None:
            for ext in PLT_SAVE_TYPES:
                savefig(save_path + ext)
        if close:
            plt.close()


class SaveResultMetric(AbstractMetric):
    def __init__(self):
        """
        DummyMetric which saves the network predictions
        Assumes that results can be pickled
        """
        super().__init__(metric_fct=None)

    def __call__(self, results, epoch, **kwargs):
        """
        Dummy
        """
        self.results = results
        self.metric[epoch] = ({'scalar': 0})
        return 0

    def log_metric(self, name, epoch, fold=1):
        """
        Dummy
        """
        pass

    def plot_metric(self, epoch, show=False, save_path=None, **kwargs):
        """
        Pickle results from previous __call__ on this metric
        """
        with open(save_path + ".pickle", 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)
