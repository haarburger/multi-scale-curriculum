import typing
import logging
from copy import deepcopy
import pandas as pd
import os

from trixi.logger.tensorboard import PytorchTensorboardXLogger
from batchgenerators.transforms import Compose
from datetime import datetime

from delira.logging import TrixiHandler
from delira.training import Parameters
from delira.data_loading.sampler import SequentialSampler, AbstractSampler
from delira.data_loading import AbstractDataset, BaseDataManager

from mscl.utils import ConfigHandlerAbstract, ConfigHandlerPyTorchDelira
from mscl.utils import SaveResultMetric


class Pipeline:
    @staticmethod
    def setup_config(paths, fcts):
        """
        Creates ConfigHandler

        Parameters
        ----------
        paths : list
            list with paths to config files
        fcts : list
            list with functions which add additional settings to config file

        Returns
        -------
        ch : ConfigHandlerPyTorchDelira
            contains all loaded configs
        """
        ch = ConfigHandlerPyTorchDelira()
        for file in paths:
            ch.load(file)

        for fct in fcts:
            ch = fct(ch)
        return ch

    @staticmethod
    def setup_logging(ch: ConfigHandlerAbstract, **kwargs):
        """
        Setup visdom logger

        Parameters
        ----------
        ch : ConfigHandlerAbstract
            contains settings for logging
            required keys: [logging]
        Returns
        -------
        logger
        """
        ch['logging.target_dir'] = os.path.join(
            ch['logging.target_dir'], str(ch['exp.name']),
            str(datetime.now().strftime("%y-%m-%d_%H-%M-%S")))

        handlers = []
        # stream_handler = logging.StreamHandler()
        # stream_handler.setLevel(logging.INFO)
        # handlers.append(stream_handler)

        if ch['logging']:
            trixi_handler = TrixiHandler(PytorchTensorboardXLogger,
                                         **ch['logging'])
            trixi_handler.setLevel(logging.INFO)
            handlers.append(trixi_handler)

        logging.basicConfig(level=logging.INFO,
                            handlers=handlers,
                            **kwargs)
        logger = logging.getLogger("Exec_Logger")
        return logger

    @staticmethod
    def setup_evaluation(ch: ConfigHandlerAbstract,
                         evaluator_cls,
                         evaluator_kwargs: dict = {},
                         batch_metrics: dict = {},
                         dataset_metrics: dict = {},
                         ):
        """
        Creates evaluators for experiment

        Parameters
        ----------
        ch : ConfigHandlerAbstract
            contains configuration settings in a dict-like object
            optinal settings used: [exp.save_train_results,
            exp.save_val_results]
        evaluator_cls :
            class of evaluator
        evaluator_kwargs :
            kwargs for evaluator
        batch_metrics :
            batch_metrics which should be added to all evaluators
        dataset_metrics :
            dataset_metrics which should be added to all evaluators

        Returns
        -------
        evaluator_dict : dict
            contains evaluators for train, val, test
        """
        evaluator_dict = {}
        dataset_metrics_train = deepcopy(dataset_metrics)
        dataset_metrics_val = deepcopy(dataset_metrics)
        dataset_metrics_test = deepcopy(dataset_metrics)

        if 'exp.save_train_results' in ch and ch['exp.save_train_results']:
            dataset_metrics_train['results'] = SaveResultMetric()

        if 'exp.save_val_results' in ch and ch['exp.save_val_results']:
            dataset_metrics_val['results'] = SaveResultMetric()

        dataset_metrics_test['results'] = SaveResultMetric()

        evaluator_dict['train'] = \
            evaluator_cls(ch=ch,
                          batch_metrics=deepcopy(batch_metrics),
                          dataset_metrics=dataset_metrics_train,
                          logger_postfix='_train',
                          **evaluator_kwargs,
                          )

        evaluator_dict['val'] = \
            evaluator_cls(ch=ch,
                          batch_metrics=deepcopy(batch_metrics),
                          dataset_metrics=dataset_metrics_val,
                          logger_postfix='_val',
                          **evaluator_kwargs,
                          )

        evaluator_dict['test'] = \
            evaluator_cls(ch=ch,
                          batch_metrics=deepcopy(batch_metrics),
                          dataset_metrics=dataset_metrics_test,
                          logger_postfix='_test',
                          **evaluator_kwargs,
                          )
        return evaluator_dict

    @staticmethod
    def setup_datasets(ch: ConfigHandlerAbstract,
                       dataset_cls: AbstractDataset,
                       load_fn,
                       path: typing.Union[list, pd.DataFrame],
                       **kwargs):
        """
        Setup train and test dataset

        Parameters
        ----------
        ch : ConfigHandlerAbstract
            provides addtional configuration for io
            required keys:
                ``data.kwargs``
            optional keys:
                ``debug_data`` reduces number of samples if ``test_path`` is 
                list
        dataset_cls : AbstractDataset
            class of dataset
        load_fn :
            load function for sample
        path : list of str or pandas.DataFrame
            if path is a list it can be split by using the confighandler option
            ``debug_data.n_train``, ``debug_data.n_val``, ``debug_data.n_test``
            where each specifies
            how many samples should be used
            if path is a pandas.DataFrame it should provide two columns: path
            and type where type should be either 'train', 'val', 'test'
        kwargs :
            variable number of keyword arguments passed to dataset

        Returns
        -------
        dict
            dict with 'train', 'val' and 'test' io if class dataset_cls


        Raises
        ------
        ValueError
            If path is not list or pandas.DataFrame
        """
        if isinstance(path, list):
            start_ind = 0

            # setup train set
            if 'debug_data.n_train' not in ch:   # only a train set
                ch['debug_data.n_train'] = len(path)
            train_path = path[start_ind:start_ind + ch['debug_data.n_train']]
            start_ind += ch['debug_data.n_train']

            # setup val set
            if 'debug_data.n_val' not in ch:   # train + val set
                ch['debug_data.n_val'] = len(path) - ch['debug_data.n_train']
            val_path = path[start_ind:start_ind + ch['debug_data.n_val']]
            start_ind += ch['debug_data.n_val']

            # setup test set
            test_path = path[start_ind:start_ind + ch['debug_data.n_test']]

        elif isinstance(path, pd.DataFrame):
            train_path = path.loc[path['type'] == 'train', 'path'].tolist()
            if 'debug_data.n_train' in ch:
                train_path = train_path[:ch['debug_data.n_train']]

            val_path = path.loc[path['type'] == 'val', 'path'].tolist()
            if 'debug_data.n_val' in ch:
                val_path = val_path[:ch['debug_data.n_val']]

            test_path = path.loc[path['type'] == 'test', 'path'].tolist()
            if 'debug_data.n_test' in ch:
                test_path = test_path[:ch['debug_data.n_test']]

        else:
            raise ValueError(f"path should be either list or pandas.DataFrame, "
                             f"not {type(path)}")

        dataset_dict = {}
        if train_path is not None and len(train_path) > 0:
            dataset_dict['train'] = dataset_cls(train_path,
                                                load_fn=load_fn,
                                                **ch['data.kwargs'],
                                                **kwargs)

        if val_path is not None and len(val_path) > 0:
            dataset_dict['val'] = dataset_cls(val_path,
                                              load_fn=load_fn,
                                              **ch['data.kwargs'],
                                              **kwargs)

        if test_path is not None and len(test_path) > 0:
            dataset_dict['test'] = dataset_cls(test_path,
                                               load_fn=load_fn,
                                               **ch['data.kwargs'],
                                               **kwargs)
        return dataset_dict

    @staticmethod
    def setup_datamanagers(ch: ConfigHandlerAbstract,
                           dataset_dict: dict,
                           datamanager_cls: BaseDataManager,
                           params: Parameters,
                           base_transforms: list,
                           train_transforms: list,
                           sampler_cls=SequentialSampler,
                           n_process_augmentation=1,
                           **kwargs):
        """
        Setup datamanagers.
        datamanager_dict['train'] contains data manager
        with base_transforms and train_transforms and uses sampler_cls
        datamanager_dict['test'] contains data manager
        with base_transforms and SequentialSampler

        Parameters
        ----------
        ch : ConfigHandlerAbstract
            contains additional configuration information

            Optional keys:
                [data.prototyping] reduces number of samples in datamanagers
        dataset_dict : dict
            dict with dataset classes. Can contain 'train' and/or 'test'
        datamanager_cls : BaseDataManager
            class for datamanager
        params : Parameters
            Parameters object which contains batch_size
        base_transforms : list
            transformations which are applied to train and test set
        train_transforms : list
            transformations which are applied only to the train set
        sampler_cls : AbstractSampler
            sampler class for train datamanager
            SequentialSampler is always used for validation
        n_process_augmentation : int
            number processes used for augmentation
        kwargs :
            variable number of keyword arguments passed through datamanagers

        Returns
        -------
        datamanager_dict : dict
            contains 'train' and 'test' datamanagers
        """
        datamanager_dict = {}
        train_transforms = Compose(base_transforms + train_transforms)
        if 'train' in dataset_dict:
            datamanager_dict['train'] = \
                datamanager_cls(dataset_dict['train'],
                                params.nested_get('batch_size'),
                                n_process_augmentation,
                                transforms=train_transforms,
                                sampler_cls=sampler_cls,
                                **kwargs,
                                )
        else:
            datamanager_dict['train'] = None

        if 'val' in dataset_dict:
            datamanager_dict['val'] = \
                datamanager_cls(dataset_dict['val'],
                                params.nested_get('batch_size'),
                                n_process_augmentation,
                                transforms=Compose(base_transforms),
                                sampler_cls=SequentialSampler,
                                **kwargs,
                                )
        else:
            datamanager_dict['val'] = None

        if 'test' in dataset_dict:
            datamanager_dict['test'] = \
                datamanager_cls(dataset_dict['test'],
                                params.nested_get('batch_size'),
                                n_process_augmentation,
                                transforms=Compose(base_transforms),
                                sampler_cls=SequentialSampler,
                                **kwargs,
                                )
        else:
            datamanager_dict['test'] = None
        return datamanager_dict
