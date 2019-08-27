import logging
import torch
import torch.nn.functional as F
from functools import partial
from delira.models.abstract_network import AbstractPyTorchNetwork
from .model import ClassModel

logger = logging.getLogger(__name__)


class ClassNetwork(AbstractPyTorchNetwork):
    """
    See Also
    --------
    :class:`AbstractPyTorchNetwork`

    """

    def __init__(self, ch, **kwargs):
        """

        Parameters
        ----------
        ch : ConfigHandler
            contains all necessary configurations

        """
        super().__init__()
        self.module = self._build_model(ch)

    def forward(self, input_batch: torch.Tensor):
        """
        Forward input_batch through network

        Parameters
        ----------
        input_batch : torch.Tensor
            batch to forward through network

        Returns
        -------
        torch.Tensor
            Classification Result

        """
        return {"pred": self.module(input_batch)}

    @staticmethod
    def closure(model: AbstractPyTorchNetwork, data_dict: dict,
                optimizers: dict, losses={}, metrics={},
                fold=0, **kwargs):
        """
        closure method to do a single backpropagation step


        Parameters
        ----------
        model : :class:`ClassificationNetworkBasePyTorch`
            trainable model
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary of optimizers to optimize model's parameters
        losses : dict
            dict holding the losses to calculate errors
            (gradients from different losses will be accumulated)
        metrics : dict
            dict holding the metrics to calculate
        fold : int
            Current Fold in Crossvalidation (default: 0)
        **kwargs:
            additional keyword arguments

        Returns
        -------
        dict
            Metric values (with same keys as input dict metrics)
        dict
            Loss values (with same keys as input dict losses)
        dict
            Arbitrary number of predictions as torch.Tensor

        Raises
        ------
        AssertionError
            if optimizers or losses are empty or the optimizers are not
            specified

        """
        assert (optimizers and losses) or not optimizers, \
            "Criterion dict cannot be emtpy, if optimizers are passed"

        loss_vals = {}
        metric_vals = {}
        total_loss = 0
        # function to compute output probability: either softmax or sigmoid

        # choose suitable context manager:
        if optimizers:
            context_man = torch.enable_grad

        else:
            context_man = torch.no_grad

        with context_man():
            # get data
            inputs = data_dict.pop('data')
            label = data_dict.pop('label', None)
            label_ohe = data_dict.pop('label_ohe', None)

            pred_dict = model(inputs)

            # print(label)
            # print(pred_dict["pred"])

            # compute loss
            for key, crit_fn in losses.items():
                _loss_val = crit_fn(pred_dict["pred"], label)
                loss_vals[key] = _loss_val.item()
                total_loss += _loss_val

        if optimizers:
            optimizers['default'].zero_grad()
            with optimizers["default"].scale_loss(total_loss) as scaled_loss:
                scaled_loss.backward()
            optimizers['default'].step()
        else:
            # add prefix "val" in validation mode
            eval_loss_vals, eval_metrics_vals = {}, {}
            for key in loss_vals.keys():
                eval_loss_vals[str(key) + '_test'] = loss_vals[key]

            loss_vals = eval_loss_vals
            metric_vals = eval_metrics_vals

        for key, val in {**metric_vals, **loss_vals}.items():
            logger.info({"value": {"value": val, "name": key,
                                   "env_appendix": "_%02d" % fold
                                   }})

        # TODO: remove
        # logging.info({
        #     'image_grid': {
        #         "image_array":
        #             inputs[:, 0, ..., inputs.shape[-1] //
        #                    2].detach().cpu().numpy(),
        #         "name": "input_images",
        #         "env_appendix": "_%02d" % fold,
        #         "normalize": True}})

        # class_pred = F.softmax(pred_dict["pred"].detach(), dim=1)

        # TODO: remove
        # # logging with evaluator
        # log_dict = {'class': {'pred': class_pred.cpu().numpy(),
        #                       'gt': label.detach().cpu().numpy()},
        #             'id': data_dict.pop('id', None),
        #             'uid': data_dict.pop('uid', None)}

        # TODO: remove
        # with torch.no_grad():
        #     if optimizers and "train" in metrics:
        #         metrics["train"](log_dict)
        #         metrics["train"].log_batch_metrics(fold=fold)
        #     elif not optimizers and "val" in metrics:
        #         metrics["val"](log_dict)
        #         metrics["val"].log_batch_metrics(fold=fold)

        return metric_vals, loss_vals, {k: v.detach().cpu().numpy() for k, v in
                                        pred_dict.items()}

    @staticmethod
    def _build_model(ch, **kwargs):
        """
        builds actual model

        Parameters
        ch : Confighandler
            provides all configuration parameters
        **kwargs : dict
            additional keyword arguments

        Returns
        -------
        torch.nn.Module
            created model

        """
        _model = ClassModel(ch)
        return _model

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device, **kwargs):
        """
        Helper Function to prepare Network Inputs and Labels (convert them to
        correct type and shape and push them to correct devices)

        Parameters
        ----------
        batch : dict
            dictionary containing all the data
        input_device : torch.device
            device for network inputs
        output_device : torch.device
            device for network outputs

        Returns
        -------
        dict
            dictionary containing data in correct type and shape and on correct
            device

        """
        out_dict = {}
        data = torch.from_numpy(batch.pop('data'))
        out_dict['data'] = data.to(input_device, dtype=torch.float)

        label = torch.from_numpy(batch.get('label'))
        out_dict['label'] = label.to(output_device, dtype=torch.long)

        if 'label_ohe' in batch:
            label_ohe = torch.from_numpy(batch.get('label_ohe'))
            out_dict['label_ohe'] = label_ohe.to(output_device,
                                                 dtype=torch.float)

        if 'id' in batch:
            out_dict['id'] = batch.get('id')

        if 'uid' in batch:
            out_dict['uid'] = batch.get('uid')

        return out_dict
