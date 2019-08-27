import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig
import logging
from functools import partial
import torch

from mscl.utils.metrics import ScalarBatchMetric, ScalarDatasetMetric, \
    LineDatasetMetric
from .losses import LogNllLoss

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, roc_auc_score, roc_curve


logger = logging.getLogger(__name__)
PLT_SAVE_TYPES = [".png", ".pdf"]

#TODO: remove


def auroc_fct(gt, pred):
    """
    Compute auroc

    Parameters
    ----------
    gt : np.ndarray
        ground truth label
    pred : np.ndarray
        network prediction

    Returns
    -------
    float
        auroc score

    See Also
    --------
    sklearn.metrics.roc_auc_score
    """
    if len(np.unique(gt)) > 1:
        auc = roc_auc_score(gt, pred)
    else:
        auc = np.array([np.nan])
    return auc


def roc_fct(gt, pred):
    """
    Compute roc curve

    Parameters
    ----------
    gt : np.ndarray
        ground truth label
    pred : np.ndarray
        network prediction

    Returns
    -------
    np.ndaray
        false positive rate
    np.ndarray
        true positive rate

    See Also
    --------
    sklearn.metrics.roc_curve
    """
    if len(np.unique(gt)) > 1:
        fpr, tpr, thresholds = roc_curve(gt, pred)
    else:
        fpr = np.array([np.nan])
        tpr = np.array([np.nan])

    return fpr, tpr


class RocMetric(LineDatasetMetric):
    def __init__(self, results_key, pred_key='pred', gt_key='gt',
                 n_classes=2, n_points=100):
        """
        Roc metric inside LineDatasetMetric for logging during training.
        Accepts nested dicts as input. If more than 2 classes are present, the 
        roc is computed for each classes independendly and (additionally) a
        micro average is computed.

        Parameters
        ----------
        results_key : hashable
            key where prediction and ground truth is located, by default None
        pred_key : hashable, optional
            key where prediction is located, by default 'pred'
        gt_key : hashable, optional
            key where ground truth is located, by default 'gt'
        n_classes : int, optional
            number of classes, by default 2
        n_points : int, optional
            number of points for interpolation (only used for logging), 
            by default 100

        Raises
        ------
        ValueError
            Roc can only be computed if at least two classes are present
        """
        super().__init__(roc_fct, results_key=results_key,
                         pre_processing=False)

        if n_classes < 2:
            raise ValueError(
                f"n_classes must be at least two, not {n_classes}")

        self.pred_key = pred_key
        self.gt_key = gt_key
        self.n_classes = n_classes
        self.n_points = n_points

    def __call__(self, results, epoch, **kwargs):
        """
        Compute metric
        """
        # concatenate targets
        gt = \
            np.concatenate([i[self.results_key][self.gt_key] for i in results])
        if self.n_classes > 2:
            # multi-class roc -> binarize labels
            gt = label_binarize(gt, range(self.n_classes))

        # concatenate predictions
        pred = \
            np.concatenate([i[self.results_key][self.pred_key]
                            for i in results])

        if self.n_classes == 2:
            # Binary classification
            fpr, tpr = self.metric_fct(gt, pred[:, 1])
            auc = auroc_fct(gt, pred[:, 1])
            self.metric[epoch] = {'prec': {'fpr': {'micro_avg': fpr},
                                           'tpr': {'micro_avg': tpr}},
                                  'auc': {'micro_avg': auc}}
            return {'ROC': self.metric[epoch]['prec']}
        else:
            # Multiple classes
            # iterpoalted values for logging
            fpr_log = np.linspace(0, 1, self.n_points)
            tpr_log = np.ndarray((self.n_points, self.n_classes))

            # precise values for plot_metric function
            fpr = {}
            tpr = {}
            auc = {}

            # compute roc curve for each class
            for i in range(self.n_classes):
                fpr[i], tpr[i] = self.metric_fct(gt[:, i], pred[:, i])
                auc[i] = auroc_fct(gt[:, i], pred[:, i])
                tpr_log[:, i] = np.interp(fpr_log, fpr[i], tpr[i])

            fpr['micro_avg'], tpr['micro_avg'] = \
                self.metric_fct(gt.ravel(), pred.ravel())
            auc['micro_avg'] = auroc_fct(gt.ravel(), pred.ravel())

            self.metric[epoch] = \
                {'log': {'fpr': np.repeat(fpr_log.reshape((-1, 1)),
                                          self.n_classes, axis=1),
                         'tpr': tpr_log},
                 'prec': {'fpr': fpr, 'tpr': tpr},
                 'auc': auc}

            return {'ROC': self.metric[epoch]['prec']}

    def log_metric(self, name, epoch, fold=1):
        """
        Log metric
        """
        if self.n_classes > 2:
            logger.info({"lineplot": {"y_vals":
                                      self.metric[epoch]['log']['tpr'],
                                      "x_vals":
                                          self.metric[epoch]['log']['fpr'],
                                      "name": name,
                                      "opts":
                                      {'legend': [f'class_{i}' for i in range(
                                                  self.n_classes)]},
                                      "env_appendix": "_%02d" % fold,
                                      "show": False,
                                      }
                         })
        else:
            logger.info({"lineplot":
                         {"y_vals":
                          self.metric[epoch]['prec']['tpr']['micro_avg'],
                          "x_vals":
                              self.metric[epoch]['prec']['fpr']['micro_avg'],
                          "name": name + '_micro_avg',
                          "env_appendix": "_%02d" % fold,
                          "show": False,
                          }
                         })

    def plot_metric(self, epoch, show=False, save_path=None, close=True,
                    *kwargs):
        """
        Save plot via matplotlib
        """
        try:
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            for key, item in self.metric[epoch]['prec']['fpr'].items():
                auc = self.metric[epoch]['auc'][key].item(0)
                plt.plot(item,
                         self.metric[epoch]['prec']['tpr'][key],
                         label=f'class_{key}: AUC {auc:0.2f}')
            plt.grid(True)
            plt.tight_layout()
            plt.legend(loc='lower right')
            plt.title('ROC curve')

            plt.xlabel('false positive rate')
            plt.ylabel('sensitivity')
            if show:
                plt.show()
            if save_path is not None:
                for ext in PLT_SAVE_TYPES:
                    savefig(save_path + ext)
            if close:
                plt.close()
        except:
            logger.warning(f'Potting failed for epoch {epoch}.')


class AurocMetric(ScalarDatasetMetric):
    def __init__(self, results_key, pred_key='pred', gt_key='gt',
                 n_classes=2, target_class=None):
        """
        Auroc metric inside ScalarDatasetMetric for logging during training.
        Accepts nested dicts as input. If more than 2 classes are present, the 
        target class must be specified for which the auroc score needs to be
        computed.

        Parameters
        ----------
        results_key : hashable
            key where prediction and ground truth is located, by default None
        pred_key : hashable, optional
            key where prediction is located, by default 'pred'
        gt_key : hashable, optional
            key where ground truth is located, by default 'gt'
        n_classes : int, optional
            number of classes, by default 2
        target_class : int, optional
            can be defined if auroc should only be computed for a single class.
            Needed if more than two classes are provided.

        Raises
        ------
        ValueError
            Roc can only be computed if at least two classes are present
        """
        super().__init__(auroc_fct, results_key=results_key,
                         pre_processing=False)

        if n_classes < 2:
            raise ValueError(
                f"n_classes must be at least two, not {n_classes}")

        self.pred_key = pred_key
        self.gt_key = gt_key
        self.n_classes = n_classes
        self.target_class = target_class

    def __call__(self, results, epoch, **kwargs):
        """
        Computes metric
        """
        # concatenate targets
        gt = \
            np.concatenate([i[self.results_key][self.gt_key] for i in results])
        if self.n_classes > 2:
            # multi-class roc -> binarize labels
            gt = label_binarize(gt, range(self.n_classes))

        # concatenate predictions
        pred = \
            np.concatenate([i[self.results_key][self.pred_key]
                            for i in results])

        if self.n_classes > 2 and (self.target_class is None):
            scalar = self.metric_fct(gt.ravel(), pred.ravel())
        elif self.n_classes > 2 and (self.target_class is not None):
            scalar = self.metric_fct(gt[:, self.target_class],
                                     pred[:, self.target_class])
        else:
            scalar = self.metric_fct(gt, pred[:, 1])
        self.metric[epoch] = ({'scalar': scalar})
        return scalar


class PrecisionMetricBatch(ScalarBatchMetric):
    def __init__(self, results_key='class', pred_key='pred', gt_key='gt',
                 **kwargs):
        """
        Precision metric per batch.

        Parameters
        ----------
        results_key: hashable
            key where prediction and ground truth is located, by default None
        pred_key: hashable, optional
            key where prediction is located, by default 'pred'
        gt_key: hashable, optional
            key where ground truth is located, by default 'gt'
        """
        self.binary = kwargs.pop('binary', 'False')
        ps = partial(precision_score, **kwargs)
        super(PrecisionMetricBatch, self).__init__(ps,
                                                   results_key,
                                                   pred_key,
                                                   gt_key)

    def __call__(self, results, epoch, **kwargs):
        """
        Compute metric
        """
        # if new epoch, create empty list
        if epoch not in self.metric:
            self.metric[epoch] = []

        # compute metric
        pred = results[self.result_key][self.pred_key]
        result = results[self.result_key][self.gt_key]

        if not self.binary:
            pred = np.argmax(pred, axis=1)
        else:
            pred = np.argmax(pred, axis=1)
            result = np.argmax(result, axis=1)

        scalar = self.metric_fct(pred, result)

        # save result
        self.metric[epoch].append({'scalar': scalar})
        return scalar


class CEMetricBatch(ScalarBatchMetric):
    def __init__(self, results_key='class', pred_key='pred', gt_key='gt',
                 **kwargs):
        """
        Compute cross entropy after softmax is applied.

        Parameters
        ----------
        results_key : hashable
            key where prediction and ground truth is located, by default None
        pred_key : hashable, optional
            key where prediction is located, by default 'pred'
        gt_key : hashable, optional
            key where ground truth is located, by default 'gt'
        """
        _fn = LogNllLoss(**kwargs)
        super().__init__(_fn, results_key, pred_key, gt_key)

    def __call__(self, results, epoch, **kwargs):
        """
        Compute metric
        """
        # if new epoch, create empty list
        if epoch not in self.metric:
            self.metric[epoch] = []

        # compute metric
        pred_np = results[self.result_key][self.pred_key]
        result_np = results[self.result_key][self.gt_key]

        pred_torch = torch.from_numpy(pred_np)
        result_torch = torch.from_numpy(result_np)

        ce_torch = self.metric_fct(pred_torch, result_torch)
        scalar = ce_torch.numpy()

        # save result
        self.metric[epoch].append({'scalar': scalar})
        return scalar
