from mscl.utils import Evaluator

from delira.training import Predictor


#TODO: remove
class TestPredictor(Predictor):
    """
    Hack to support Evaluator class to compute metrics over dataset
    (feature is currently under development in Delira repo and will be updated
    when finished)
    """
    @staticmethod
    def calc_metrics(batch: dict, metrics={}, metric_keys=None):
        """
        Compute metrics with evaluator from metrics dict

        Parameters
        ----------
        batch : dict
            dict containing batch data and predictions
        metrics : dict, optional
            dict which contains evaluator, by default {}
        metric_keys : dict, optional
            map batch keys to new dict which is passed to evaluator
            (key specifies key in batch, item specifies key in new dict)
        """
        # create remapped dict
        log_dict = {}
        for key, item in batch.items():
            if key in metric_keys:
                log_dict[metric_keys[key]] = item
            else:
                log_dict[key] = item
        log_dict['class'] = {'gt': log_dict['gt'],
                             'pred': log_dict['pred']}

        # compute batch metric
        for key, item in metrics.items():
            assert isinstance(item, Evaluator)

            item(log_dict)
            item.log_batch_metrics()
        return {}
