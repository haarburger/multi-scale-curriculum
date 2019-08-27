from abc import abstractmethod
from trixi.util import Config

import warnings
import os
from copy import deepcopy

from yaml import load, dump, YAMLError
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import torch
from delira.training import Parameters
import delira


class ConfigHandlerAbstract(Config):
    def __init__(self, *args, **kwargs):
        """
        Extends trixi config by additional formats like yaml and provides some
        framework specific functions
        """
        super().__init__(*args, **kwargs)

    @abstractmethod
    def load(self, path, *args, **kwargs):
        """
        Loads settings from config file. Can load new file formats or uses
        original load function to load json

        Parameters
        ----------
        path : string
            path to config file
        args :
            variable number of arguments passed to load function
        kwargs :
            variable number of keyword arguments passed to load function
        """
        super().load(path, *args, **kwargs)

    def __contains__(self, item, superkey=''):
        if '.' not in item:
            return super().__contains__(item)
        key_list = item.split('.')
        if key_list[0] in (self[superkey] if superkey else self):
            if superkey:
                superkey = superkey + '.' + key_list[0]
            else:
                superkey = key_list[0]
            if len(key_list) == 2:
                if isinstance(self[superkey], dict) and \
                        key_list[1] in self[superkey]:
                    return True
                else:
                    return False
            else:
                return self.__contains__('.'.join(key_list[1:]), superkey)
        else:
            return False


class ConfigHandlerYAML(ConfigHandlerAbstract):
    def __init__(self, *args, **kwargs):
        """
        ConfigHandler for YAML files
        Special keys:
            'config_list': loads additional configs from list
            'config_dict': loads additional configs from dict an saves
            each config in the corresponding key
        """
        super().__init__(*args, **kwargs)

    def load(self, path, *args, in_key=None, **kwargs):
        """
        Can load either YAML or JSON file. Special keys only work for yml file

        Parameters
        ----------
        path : string
            Path to YAML file
        in_key : string
            specifies a key where to save new config file
        args :
            variable number of arguments passed to load function
        kwargs : 
            variable number of keyword arguments passed to load function

        Raises
        ------
        ValueError:
            if target file is not yaml, yml or json
        """
        basename = os.path.basename(path)
        basename = basename.split('.')
        # load yaml file
        if basename[1] in ['yml', 'yaml']:
            with open(path, 'r') as stream:
                try:
                    data = load(stream, *args, **kwargs)
                except YAMLError as exc:
                    warnings.warn(exc)
                    return

            config_list = data.pop('config_list', None)
            config_dict = data.pop('config_dict', None)

            for key, item in data.items():
                if in_key is None:
                    self[key] = item
                else:
                    self[in_key + '.' + str(key)] = item

            if config_list is not None:
                if not isinstance(config_list, list):
                    raise ValueError("config_list must be a list!")
                for val in config_list:
                    self.load_config(val)

            if config_dict is not None:
                if not isinstance(config_dict, dict):
                    raise ValueError("config_dict must be a dict!")
                for key, val in config_dict.items():
                    self.load_config(val, in_key=str(key))
        # load json
        elif basename[1] in ['json']:
            super().load(path, *args, **kwargs)
        # raise error
        else:
            raise ValueError(f"File format {basename[1]} is not supported. "
                             f"Supported formats are yaml, yml, json.")


class ConfigHandlerPyTorch(ConfigHandlerYAML):
    def __init__(self, *args, **kwargs):
        """
       ConfigHandlerYAML with additional functionality for PyTorch
       """
        # Read yaml file
        super().__init__(*args, **kwargs)

        self['_internal_fct'] = {}
        self.optim_source = torch.optim
        self.scheduler_source = torch.optim.lr_scheduler

    def dump(self, *args, **kwargs):
        super().dump(*args, **kwargs)

    def get_optimizer(self, optimizer_string=None):
        """
        Selects the optimizer defined in data
        Note: not all optimizers are supported

        Returns
        -------
            pytorch optimizer class
        """
        if optimizer_string is not None:
            return getattr(self.optim_source, optimizer_string)
        elif ('optimizer' in self) and (self['optimizer'] is not None):
            return getattr(self.optim_source, self['optimizer'])

    def get_scheduler(self, scheduler_string=None):
        """
        Selects the scheduler defined in data#
        Note: not all schedulers are supported

        Returns
        -------
            pytorch scheduler class
        """
        if scheduler_string is not None:
            return getattr(self.optim_source, scheduler_string)
        elif ('scheduler' in self) and (self['scheduler'] is not None):
            return getattr(self.optim_source, self['scheduler'])

    def save(self, filepath: str):
        """
        Remove modules before saving becasue they can not be pickled

        Parameters
        ----------
        filepath : str
            path where file should be saved
        """
        # backup module sources because they can not be pickled
        optim_source_ref = self.optim_source
        scheduler_source_ref = self.scheduler_source
        delattr(self, "optim_source")
        delattr(self, "scheduler_source")

        super().save(filepath)

        self.optim_source = optim_source_ref
        self.scheduler_source = scheduler_source_ref


class ConfigHandlerPyTorchDelira(ConfigHandlerPyTorch):
    def __init__(self, *args, **kwargs):
        """
        ConfigHandlerYAML with additional functionality for Delira
        """
        # Read yaml file
        super().__init__(*args, **kwargs)

        # Supprted delira scheduler callbacks
        self.scheduler_source = delira.training.callbacks.pytorch_schedulers

    def get_params(self, losses: dict, train_metrics: dict,
                   val_metrics: dict, add_self=True):
        """
        Parses all necessary information from config file to generate
        a `Delira` params dict

        Parameters
        ----------
        losses : dict
            dict containing losses
        metrics : dict
            dict containing metrics
        add_self : bool
            adds settings from confighandler to parameters object in
            ``fixed_params.model.ch``

        Returns
        -------
        dict:
            can be passed directly to Parameters class from Delira
        """
        if 'parameters' in self:
            # pop parameters from ConfigHandler due to problems with
            # lookup in parameters object
            params = deepcopy(self['parameters'])
            params['fixed_params']['training']['optimizer_cls'] = \
                self.get_optimizer(params['fixed_params']
                                   ['training']['optimizer_cls'])
            params['fixed_params']['training']['lr_sched_cls'] = \
                self.get_scheduler(params['fixed_params']
                                   ['training']['lr_sched_cls'])
            params['fixed_params.training.train_metrics'] = train_metrics
            params['fixed_params.training.val_metrics'] = val_metrics
            params['fixed_params.training.losses'] = losses

            if add_self:
                # backup module sources because they can not be pickled
                optim_source_ref = self.optim_source
                scheduler_source_ref = self.scheduler_source
                delattr(self, "optim_source")
                delattr(self, "scheduler_source")

                # pop 'parameter' from config before adding
                # otherwise `nested_get` would not work properly
                tmp = deepcopy(self)
                tmp.pop('parameters')
                params['fixed_params.model.ch'] = tmp

                self.optim_source = optim_source_ref
                self.scheduler_source = scheduler_source_ref
        else:
            optimizer_cls = self.get_optimizer()
            scheduler_cls = self.get_scheduler()
            fixed_params = {
                "model": {
                    **self['model_kwargs']
                },
                "training": {
                    'batch_size': self['batchsize'],
                    'num_epochs': self['num_epochs'],
                    'optimizer_cls': optimizer_cls,
                    'optimizer_params': self['optimizer_kwargs'],
                    'losses': losses,
                    'lr_sched_cls': scheduler_cls,
                    'lr_sched_params': self['scheduler_kwargs'],
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                }
            }
            params = {'fixed_params': fixed_params}
            if add_self:
                raise NotImplementedError("Add_self option is not implemented"
                                          "if parameters are not defined in "
                                          "config")
        return params


def feature_map_params(ch: ConfigHandlerAbstract):
    """
    Compute number of channels and size of feature map for every resolution
    stage with regard to the selected backbone architecture

    Parameters
    ----------
    ch : ConfigHandlerAbstract
        dict like object which contains all necessary configurations

    Returns
    -------
    ConfigHandlerAbstract
        new handler with additional configs
    """
    import numpy as np

    def resnet_config(ch: ConfigHandlerAbstract):
        from mscl.models.resnet import RESNETS

        # compute strides
        reduced_pool = ch['reduced_pool'] if 'reduced_pool' in ch else False
        operate_stride1 = False
        dim = ch['dim']

        xy = [2, 4, 8, 16, 32]

        if reduced_pool:
            z = [1, 1, 1, 2, 4]
        else:
            z = [1, 1, 2, 4, 8]

        if operate_stride1:
            xy = [1] + xy
            z = [1] + z

        if dim == 2:
            strides = {'xy': xy}
        else:
            strides = {'xy': xy, 'z': z}

        # save strides in confighandler
        ch['backbone_strides'] = strides

        # compute number of filters for each resolution
        start_filts = ch['start_filts']
        operate_stride1 = False
        block_expansion = RESNETS[ch['architecture']]['expansion']
        blocks = len(RESNETS[ch['architecture']]['block_list'])

        filts = []
        if operate_stride1:
            filts.append(start_filts)
        filts.append(start_filts)

        filts += [start_filts * block_expansion * 2 ** ii for ii in
                  range(0, blocks)]
        # save filts in confighandler
        ch['filts'] = filts

        return ch

    def densenet_config(ch: ConfigHandlerAbstract):
        from mscl.models.densenet import DENSENETS
        # compute strides
        reduced_pool = ch['reduced_pool'] if 'reduced_pool' in ch else False
        operate_stride1 = False
        out_trans = ch['out_trans'] if 'out_trans' in ch else True
        dim = ch['dim']

        if out_trans:
            xy = [4, 8, 16, 32, 64]
        else:
            xy = [2, 4, 8, 16, 64]

        if reduced_pool:
            if out_trans:
                z = [1, 1, 1, 2, 4]
            else:
                z = [1, 1, 1, 1, 4]
        else:
            if out_trans:
                z = [1, 1, 2, 4, 8]
            else:
                z = [1, 1, 1, 2, 8]

        if operate_stride1:
            xy = [1] + xy
            z = [1] + z

        if dim == 2:
            strides = {'xy': xy}
        else:
            strides = {'xy': xy, 'z': z}

        # save strides in confighandler
        ch['backbone_strides'] = strides

        # compute number of filters for each resolution
        block_config = DENSENETS[ch['architecture']]['block_list']
        growth_rate = DENSENETS[ch['architecture']]['growth_rate']
        num_init_features = \
            DENSENETS[ch['architecture']]['num_init_features']
        out_trans = ch['out_trans'] if 'out_trans' in ch else True

        if 'start_filts' in ch:
            num_init_features = ch['start_filts']

        if 'growth_rate' in ch:
            growth_rate = ch['growth_rate']
        operate_stride1 = ch['operate_stride1']

        filts = []
        if operate_stride1:
            filts.append(num_init_features)
        filts.append(num_init_features)

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            num_features = num_features + num_layers * growth_rate

            if not out_trans:
                filts.append(num_features)

            # correct for last dense block without transition layer
            if i != len(block_config) - 1:
                num_features = num_features // 2

            if out_trans:
                filts.append(num_features)
        # save filts in confighandler
        ch['filts'] = filts

        return ch

    BACKS = {
        'resnet': resnet_config,
        'densenet': densenet_config,
    }

    # determine channels and strides
    config_fn = None
    for key, item in BACKS.items():
        if key in ch['architecture']:
            config_fn = item
    if config_fn is None:
        raise ValueError("Backbone {} is not supported.".format(
            ch['architecture']))
    ch = config_fn(ch)

    # compute size of feature maps
    if ch['dim'] == 2:
        ch['backbone_shapes'] = np.array(
            [[int(np.ceil(ch['patch_size'][0] / stride)),
              int(np.ceil(ch['patch_size'][1] / stride))]
             for stride in ch['backbone_strides']['xy']])
    else:
        ch['backbone_shapes'] = np.array(
            [[int(np.ceil(ch['patch_size'][0] / stride)),
              int(np.ceil(ch['patch_size'][1] / stride)),
              int(np.ceil(ch['patch_size'][2] / stride_z))]
             for stride, stride_z in zip(ch['backbone_strides']['xy'],
                                         ch['backbone_strides']['z'])])

    return ch
