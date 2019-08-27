import typing
import torch
import torch.nn as nn
import numpy as np
import copy
import logging
from collections import OrderedDict
from delira.io import torch_load_checkpoint
from mscl.utils import ConfigHandlerAbstract, NDConvGenerator, \
    ConfigHandlerYAML, WeightInitializer, ndpool
from mscl.models.resnet import ResNetBackbone

logger = logging.getLogger(__name__)

BACKS = {
    'resnet': ResNetBackbone
}


def get_back(ch: ConfigHandlerAbstract):
    """
    Determine class of current backbone

    Parameters
    ----------
    ch : ConfigHandlerAbstract
        dict like object containing all configurations. `architecture` key
        determined the backgbone network
        Required keys:
            architecture : str
                define backbone class

    Returns
    -------
    torch.nn.Module
        network class

    Raises
    ------
    ValueError
        raised if `BACKS` contains ambigious keys
    ValueError
        raised if `architecture` in ch is not matching any backbone

    See Also
    --------
    :class:`ConfigHandlerAbstract`
    """
    back_cls = None
    for key, item in BACKS.items():
        if key in ch['architecture']:
            if back_cls is None:
                back_cls = item
            else:
                raise ValueError(f"architecture /'{ch['architecture']}"
                                 f"/' is ambiguous!")
    if back_cls is None:
        raise ValueError("Backbone {} is not supported.".format(
            ch['architecture']))
    return back_cls


def build_back(ch: ConfigHandlerAbstract, conv: NDConvGenerator):
    """
    Create instance of backbone network

    Parameters
    ----------
    ch : :class:`ConfigHandlerAbstract`
        dict like object containing necessary configurations
        Required keys:
            architecture : str
                define backbone class
    conv : :class:`NDConvGenerator`
        convolution wrapper to support 2d and 3d

    Returns
    -------
    torch.nn.Module
        torch model

    See Also
    --------
    :func:`get_back`, :class:`NDConvGenerator`, :class:`ConfigHandlerAbstract`
    """
    return get_back(ch)(ch, conv)


class ClassModel(nn.Module):
    def __init__(self, ch: ConfigHandlerAbstract, **kwargs):
        """
        Build different classification networks and provide functionality
        to load old weights or initiate next training stage with different
        resolutions

        Parameters
        ----------
        ch : ConfigHandlerAbstract
            contains all necessary information for later functions
            Required keys:
                dim: int
                    dimensionality of input (either 2 or 3)
            Optional keys:
                load.config: str
                    path to old config file
                load.weights: str
                    path to weights which should be loaded
                patchsize: list of int
                    patch size of input, used to determine scale of new feature
                    maps (if different from old patch size)
                weight_init: str
                    specify the initialization method. 'xavier_uniform'|
                    'xavier_normal'|'kaiming_uniform'|'kaiming_normal'.
                    Default: 'kaiming_uniform'
                ms_pooling : bool
                    adds an adaptive pooling layer inside
                    :class:`MultiScaleHead`
                relu: str
                    string which specifies non-linearity
        kwargs :
            variable number of keyword arguments passed to :func:`_build_model`

        See Also
        --------
        :class:`WeightInitializer`, :class:`NDConvGenerator`
        """
        super().__init__()
        self.model = self._build_model(ch, **kwargs)

    def forward(self, input):
        """
        Forward input trhough network
        Parameters
        ----------
        input : torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            classification result
        """
        return self.model(input)

    def _build_model(self, ch: ConfigHandlerAbstract, **kwargs):
        """
        Build model specified by config file
        Parameters
        ----------
        ch : :class:`ConfigHandlerAbstract`
            contains all necessary information, see ``__init__``
        kwargs :
            variable number of keyword arguments

        Returns
        -------
        torch.nn.Module
            classification model
        """
        conv = NDConvGenerator(ch['dim'])

        # load old config file if possible
        if ('load' in ch) and ch['load']:
            if 'load.config' in ch:
                ch_old = ConfigHandlerYAML()
                ch_old.load(ch['load.config'])
            else:
                logger.info('No config file found in load.')
                ch_old = ch
        else:
            ch_old = ch

        # create ResNet model
        model = ResNet(ch_old, conv)

        # initialize weights of network
        init_mode = ch['weight_init'] if 'weight_init' in ch \
            else 'kaiming_uniform'
        initializer = WeightInitializer(init_mode, relu=ch['relu'])
        initializer(net=model)

        # load weights for network
        if 'load.weights' in ch:
            checkpoint = torch_load_checkpoint(ch['load.weights'])

            state = OrderedDict()
            for key, item in checkpoint['model'].items():
                if 'model' in key:
                    substrings = str(copy.copy(key)).split('.')
                    ind = substrings.index('model')
                    state['.'.join(substrings[ind + 1:])] = \
                        checkpoint['model'][key]
            model.load_state_dict(state)

        if 'load.config' in ch:
            # compute size of new output layer
            if 'patch_size' in ch:
                new_size = np.divide(ch['patch_size'],
                                     ch_old['patch_size']).astype(np.int)

                # update out_layer for new resolution
                pooling = ch['ms_pooling'] if 'ms_pooling' in ch else False
                model.layer_out = MultiScaleHead(model.channel_total,
                                                 model.num_classes,
                                                 shape=tuple(new_size),
                                                 conv=conv,
                                                 pooling=pooling,
                                                 )
        # model is put to gpu after creation
        return model.to('cpu')


class MultiScaleHead(nn.Module):
    def __init__(self, channels_in: int, channels_out: int,
                 shape: typing.Union[tuple, int], conv: NDConvGenerator,
                 pooling=False, *args, **kwargs):
        """
        Constructs the network output

        Parameters
        ----------
        channels_in : int
            number of input channels
        channels_out : int
            number of output channels
        shape : tuple of int or int
            spatial shape of input (without batch and channel dim)
        conv : NDConvGenerator
            convolutional interface which should be used
        pooling : bool
            insert an adaptive pooling layer before the convolution

        See Also
        --------
        :class:`NDConvGenerator`, :function:`ndpool`
        """
        super().__init__()
        self.pooling = pooling

        if self.pooling:
            # adaptive pooling into fully connected layer
            out_size = (1, 1) if conv.dim == 2 else (1, 1, 1)
            self.conv = nn.Sequential(
                ndpool(conv.dim, mode='avg', output_size=out_size,
                       adaptive=True),
                # simulate fully connected layer with convolution
                conv(channels_in, channels_out, ks=1, pad=0, stride=1,
                     norm=None, relu=None)
            )
        else:
            # convolution with kernel size as big as input shape
            # => simulates fc layer
            self.conv = conv(channels_in, channels_out, ks=shape,
                             pad=0, stride=1, norm=None, relu=None)

    def forward(self, input):
        """
        Forward input trough layers

        Parameters
        ----------
        input : torch.Tensor
            torch.Tensor witch channels_in channels

        Returns
        -------
        torch.Tensor
            tensor with channels_out channels
        """
        return self.conv(input)


class ResNet(nn.Module):
    def __init__(self, ch: ConfigHandlerAbstract, conv: NDConvGenerator):
        """
        Defines fully convolutional ResNet (linear output layer is replaced by
        a convolution)

        Parameters
        ----------
        ch : :class:`ConfigHandlerAbstract`
            contains all needed settings
            Required keys:
                operate_stride1 : bool
                    additional layer at the beginning of the ResNet
                backbone_shapes : list of tuple of int
                    defines the feature map shapes from the backbone network
                filts : list of int
                    number of channels per feature map from backbone network
                head_classes : int
                    number of output classes
            Optional keys:
                dropout : float
                    dropout rate used before output layer
                ms_pooling : bool
                    adds an adaptive pooling layer inside
                    :class:`MultiScaleHead`
        conv : :class:`NDConvGenerator`
            convolutional interface which should be used

        See Also
        --------
        :class:`ConfigHandlerAbstract` :class:`NDConvGenerator`,
        :function:`ndpool`
        """
        super().__init__()

        # generate backbone
        self.operate_stride1 = False
        self.dropout = ch['dropout'] if 'dropout' in ch else None
        self.channel_total = ch['filts'][-1]
        self.num_classes = ch['head_classes']
        self.backbone = build_back(ch, conv)

        # last pooling layer
        self.global_pool = ndpool(conv.dim, 'avg',
                                  kernel_size=tuple(ch['backbone_shapes'][-1]))
        if self.dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout, inplace=True)

        # convolutional layer (equivalent to fully connected layer
        # most networks do not use the option below (squeezenet uses out_pool)
        self.layer_out = MultiScaleHead(self.channel_total,
                                        self.num_classes,
                                        shape=1, conv=conv,
                                        )

    def forward(self, input: torch.Tensor):
        """
        Forward input trough layers

        Parameters
        ----------
        input : torch.Tensor
            input tensor for ResNet

        Returns
        -------
        torch.Tensor
            tensor with head_classes channels
        """
        feature_maps = self.backbone(input)

        # only use last feature map for resnet
        out = self.global_pool(feature_maps[-1])

        if self.dropout:
            out = self.dropout_layer(out)

        # fully connected layer
        out = self.layer_out(out)
        out = out.view(out.size(0), -1)
        return out
