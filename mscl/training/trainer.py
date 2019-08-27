from delira.training import PyTorchNetworkTrainer

#TODO: remove
class PyTorchNetworkTrainerEvaluator(PyTorchNetworkTrainer):
    def __init__(self, network, save_path, key_mapping,
                 evaluator_train, evaluator_val,
                 losses=None,
                 optimizer_cls=None, optimizer_params={}, train_metrics={},
                 val_metrics={}, lr_scheduler_cls=None, lr_scheduler_params={},
                 gpu_ids=[], save_freq=1, optim_fn=create_optims_default,
                 logging_type='tensorboardx', logging_kwargs={},
                 fold=0, callbacks=[], start_epoch=1, metric_keys=None,
                 convert_batch_to_npy_fn=convert_torch_tensor_to_npy,
                 mixed_precision=False,
                 mixed_precision_kwargs={'enable_caching': True,
                                         'verbose': False,
                                         'allow_banned': False},
                 criterions=None, val_freq=1,
                 **kwargs):
        """
        Same functionality as PyTorchNetworkTrainer with additional
        Evaluator support to compute metric over entire dataset

        Parameters:
        evaluator_train : Evaluator
            evaluator for training data
        evaluator_val: Evaluator
            evaluator for validation data
        """
        self.evaluator_train = evaluator_train
        self.evaluator_val = evaluator_val
        super().__init__(network, save_path, key_mapping, losses=losses,
                         optimizer_cls=optimizer_cls,
                         optimizer_params=optimizer_params,
                         train_metrics=train_metrics,
                         val_metrics=val_metrics,
                         lr_scheduler_cls=lr_scheduler_cls,
                         lr_scheduler_params=lr_scheduler_params,
                         gpu_ids=gpu_ids,
                         save_freq=save_freq, optim_fn=optim_fn,
                         logging_type=logging_type,
                         logging_kwargs=logging_kwargs,
                         fold=fold, callbacks=callbacks,
                         start_epoch=start_epoch,
                         metric_keys=metric_keys,
                         convert_batch_to_npy_fn=convert_batch_to_npy_fn,
                         mixed_precision=mixed_precision,
                         mixed_precision_kwargs=mixed_precision_kwargs,
                         criterions=criterions, val_freq=val_freq,
                         **kwargs)
