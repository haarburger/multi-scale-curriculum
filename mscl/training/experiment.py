from functools import partial

from delira.training import Predictor
from delira.training import PyTorchExperiment
from delira.training.train_utils import convert_torch_tensor_to_npy

#TODO: remove
class PytorchExperimentPredictor(PyTorchExperiment):
    """
    Change predictor during testing
    """

    def _setup_test(self, params, model, convert_batch_to_npy_fn,
                    prepare_batch_fn, predictor_cls=Predictor, **kwargs):
        predictor = predictor_cls(
            model=model, key_mapping=self.key_mapping,
            convert_batch_to_npy_fn=convert_batch_to_npy_fn,
            prepare_batch_fn=prepare_batch_fn, **kwargs)
        return predictor
