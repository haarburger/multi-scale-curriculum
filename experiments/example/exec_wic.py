from mscl.utils import ConfigHandlerPyTorchDelira, ConfigHandlerAbstract, \
    feature_map_params
from mscl.training.wrapper import metric_wrapper_pytorch
from mscl.models import ClassNetwork
from mscl.io import load_pickle, PopKeys
import delira

from delira.data_loading import LoadSampleLabel
from delira.training.train_utils import create_optims_default_pytorch
from delira.training import Parameters, PyTorchNetworkTrainer, \
    PyTorchExperiment
from delira.data_loading.sampler import WeightedPrevalenceRandomSampler, \
    SequentialSampler
from delira.data_loading import BaseDataManager, BaseCacheDataset, \
    BaseLazyDataset
from batchgenerators.transforms import SpatialTransform, MirrorTransform, \
    Compose

from sklearn.metrics import roc_curve, auc
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
import os
import argparse
import logging
import sys


logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("Execute_Logger")


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(0)
np.random.seed(0)


def run_experiment(cp: str, test=True) -> str:
    """
    Run classification experiment on patches
    Imports moved inside because of logging setups

    Parameters
    ----------
    ch : str
        path to config file
    test : bool
        test best model on test set

    Returns
    -------
    str
        path to experiment folder
    """
    # setup config
    ch = ConfigHandlerPyTorchDelira(cp)
    ch = feature_map_params(ch)

    if 'mixed_precision' not in ch or ch['mixed_precision'] is None:
        ch['mixed_precision'] = True
    if 'debug_delira' in ch and ch['debug_delira'] is not None:
        delira.set_debug_mode(ch['debug_delira'])
        print("Debug mode active: settings n_process_augmentation to 1!")
        ch['augment.n_process'] = 1

    dset_keys = ['train', 'val', 'test']

    losses = {'class_ce': torch.nn.CrossEntropyLoss()}
    train_metrics = {}
    val_metrics = {'CE': metric_wrapper_pytorch(torch.nn.CrossEntropyLoss())}
    test_metrics = {'CE': metric_wrapper_pytorch(torch.nn.CrossEntropyLoss())}

    #########################
    #   Setup Parameters    #
    #########################
    params_dict = ch.get_params(losses=losses,
                                train_metrics=train_metrics,
                                val_metrics=val_metrics,
                                add_self=ch['add_config_to_params'])
    params = Parameters(**params_dict)

    #################
    #   Setup IO    #
    #################
    # setup io
    datasets = {}
    for key in dset_keys:
        p = os.path.join(ch["data.path"], str(key))

        datasets[key] = BaseCacheDataset(
            p, load_fn=load_pickle, **ch['data.kwargs'])

    #############################
    #   Setup Transformations   #
    #############################
    base_transforms = []
    base_transforms.append(PopKeys("mapping"))

    train_transforms = []
    if ch['augment.mode']:
        logger.info("Training augmentation enabled.")
        train_transforms.append(
            SpatialTransform(patch_size=ch['patch_size'],
                             **ch['augment.kwargs']))
        train_transforms.append(MirrorTransform(axes=(0, 1)))
    process = ch['augment.n_process'] if 'augment.n_process' in ch else 1

    #########################
    #   Setup Datamanagers  #
    #########################
    datamanagers = {}
    for key in dset_keys:
        if key == 'train':
            trafos = base_transforms + train_transforms
            sampler = WeightedPrevalenceRandomSampler
        else:
            trafos = base_transforms
            sampler = SequentialSampler

        datamanagers[key] = BaseDataManager(
            data=datasets[key],
            batch_size=params.nested_get('batch_size'),
            n_process_augmentation=process,
            transforms=Compose(trafos),
            sampler_cls=sampler,
        )

    #############################
    #   Initialize Experiment   #
    #############################
    experiment = \
        PyTorchExperiment(
            params=params,
            model_cls=ClassNetwork,
            name=ch['exp.name'],
            save_path=ch['exp.dir'],
            optim_builder=create_optims_default_pytorch,
            trainer_cls=PyTorchNetworkTrainer,
            mixed_precision=ch['mixed_precision'],
            mixed_precision_kwargs={'verbose': False},
            key_mapping={"input_batch": "data"},
            **ch['exp.kwargs'],
        )

    # save configurations
    ch.dump(os.path.join(experiment.save_path, 'config.json'))

    #################
    #   Training    #
    #################
    model = experiment.run(datamanagers['train'],
                           datamanagers['val'],
                           save_path_exp=experiment.save_path,
                           ch=ch,
                           metric_keys={'val_CE': ['pred', 'label']},
                           val_freq=1,
                           verbose=True)
    ################
    #   Testing    #
    ################
    if test and datamanagers['test'] is not None:
        # metrics and metric_keys are used differently than in original
        # Delira implementation in order to support Evaluator
        # see mscl.training.predictor
        preds = experiment.test(network=model,
                                test_data=datamanagers['test'],
                                metrics=test_metrics,
                                metric_keys={'CE': ['pred', 'label']},
                                verbose=True,
                                )

        softmax_fn = metric_wrapper_pytorch(
            partial(torch.nn.functional.softmax, dim=1))
        preds = softmax_fn(preds[0]['pred'])
        labels = [d['label'] for d in datasets['test']]
        fpr, tpr, thresholds = roc_curve(labels, preds[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label='ROC (AUC = %0.2f)' % roc_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(experiment.save_path, 'test_roc.pdf'))
        plt.close()

        preds = experiment.test(network=model,
                                test_data=datamanagers['val'],
                                metrics=test_metrics,
                                metric_keys={'CE': ['pred', 'label']},
                                verbose=True,
                                )

        preds = softmax_fn(preds[0]['pred'])
        labels = [d['label'] for d in datasets['val']]
        fpr, tpr, thresholds = roc_curve(labels, preds[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label='ROC (AUC = %0.2f)' % roc_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(experiment.save_path, 'best_val_roc.pdf'))
        plt.close()

    return experiment.save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run experiment")
    parser.add_argument('-cp', '--config_path', required=True,
                        help='Path to experiment specific config file',
                        type=str)
    args = vars(parser.parse_args())
    cp = args.get("config_path")
    experiment_path = run_experiment(cp, test=True)
