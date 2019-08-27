import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__file__)


class NDConvGenerator(object):
    def __init__(self, dim):
        """
        Wrapper for 2d/3d convolutions and relu
        Slightly modified version of medical detection toolkit
        https://github.com/pfjaeger/medicaldetectiontoolkit/blob/master/utils/model_utils.py

        Parameters
        ----------
        dim : int
            input dimensionality: either 2d or 3d
        """
        self.dim = dim

    def __call__(self, c_in, c_out, ks, pad=0, stride=1, norm=None,
                 relu='relu', bias=True, pre_act=False):
        """
        Creates a new nd layer

        Parameters
        ----------
        c_in : int
            number of in_channels
        c_out : int
            number of out_channels
        ks : int or tuple
            kernel size
        pad : int or tuple, optional
            pad size. Default: 0
        stride : int or tuple, optional
            kernel stride. Default: 1
        norm : str, optional
            string specifying type of feature map normalization. If None, no
            normalization is applied.
            Supports: 'batch_norm', 'instance_norm', None. Default: None
        relu : str, optional
            string specifying type of nonlinearity. If None, no nonlinearity is
            applied. Supports: 'relu', 'leaky_relu', None. Default: 'relu'
        bias : bool
            bias for convolution
        pre_act : bool
            pre activation changes the order to norm->relu->conv
        Returns
        -------
        torch.nn.Sequential
            conv layer with optional relu and norm
        """
        # code duplication to retain backwards compatibility
        # no pre-activation
        if not pre_act:
            if self.dim == 2:
                conv = nn.Conv2d(c_in, c_out, kernel_size=ks,
                                 padding=pad, stride=stride, bias=bias)
                if norm is not None:
                    if norm == 'instance_norm':
                        norm_layer = nn.InstanceNorm2d(c_out)
                    elif norm == 'batch_norm':
                        norm_layer = nn.BatchNorm2d(c_out)
                    else:
                        raise ValueError(
                            'norm type as specified in configs is '
                            'not implemented...')
                    conv = nn.Sequential(conv, norm_layer)

            else:
                conv = nn.Conv3d(c_in, c_out, kernel_size=ks,
                                 padding=pad, stride=stride, bias=bias)
                if norm is not None:
                    if norm == 'instance_norm':
                        norm_layer = nn.InstanceNorm3d(c_out)
                    elif norm == 'batch_norm':
                        norm_layer = nn.BatchNorm3d(c_out)
                    else:
                        raise ValueError(
                            'norm type as specified in configs is not '
                            'implemented... {}'.format(norm))
                    conv = nn.Sequential(conv, norm_layer)

            if relu is not None:
                if relu == 'relu':
                    relu_layer = nn.ReLU(inplace=True)
                elif relu == 'leaky_relu':
                    relu_layer = nn.LeakyReLU(inplace=True)
                else:
                    raise ValueError(
                        'relu type as specified in configs is '
                        'not implemented...')
                conv = nn.Sequential(conv, relu_layer)
        else:
            # pre-activation
            layers = []
            if norm is not None:
                if norm == 'instance_norm':
                    if self.dim == 2:
                        norm_layer = nn.InstanceNorm2d(c_in)
                    else:
                        norm_layer = nn.InstanceNorm3d(c_in)
                elif norm == 'batch_norm':
                    if self.dim == 2:
                        norm_layer = nn.BatchNorm2d(c_in)
                    else:
                        norm_layer = nn.BatchNorm3d(c_in)
                else:
                    raise ValueError(
                        'norm type as specified in configs is '
                        'not implemented...')
                layers.append(norm_layer)

            if relu is not None:
                if relu == 'relu':
                    relu_layer = nn.ReLU(inplace=True)
                elif relu == 'leaky_relu':
                    relu_layer = nn.LeakyReLU(inplace=True)
                else:
                    raise ValueError(
                        'relu type as specified in configs is '
                        'not implemented...')
                layers.append(relu_layer)

            if self.dim == 2:
                conv_layer = nn.Conv2d(c_in, c_out, kernel_size=ks,
                                       padding=pad, stride=stride, bias=bias)
            else:
                conv_layer = nn.Conv3d(c_in, c_out, kernel_size=ks,
                                       padding=pad, stride=stride, bias=bias)
            layers.append(conv_layer)
            conv = nn.Sequential(*layers)
        return conv


def ndpool(dim, mode, adaptive=False, **kwargs):
    """
    Provides a wrapper for 2d/3d pooling layers for code reduction

    Parameters
    ----------
    dim : int
        2d or 3d supported
    mode : str
        either 'max' or 'avg' for MaxPooling or AveragePooling.
        Also supported are 'adaptive_max' or 'adaptive_avg' for the
        respective adptive version (alternatively this can also be
        enabled with the 'adaptive' keyword)
    adaptive : bool
        changes to adaptive pooling
    kwargs :
        additional keyword arguments passed to pooling layer

    Returns
    -------
    torch.nn.Module
        defined pooling layer
    """
    if dim == 2:
        if 'max' in mode:
            if adaptive or mode == 'adaptive_max':
                return nn.AdaptiveMaxPool2d(**kwargs)
            else:
                return nn.MaxPool2d(**kwargs)
        elif 'avg' in mode:
            if adaptive or mode == 'adaptive_avg':
                return nn.AdaptiveAvgPool2d(**kwargs)
            else:
                return nn.AvgPool2d(**kwargs)
        else:
            raise ValueError(f'Only "max" or "avg" supported, not {mode}')
    elif dim == 3:
        if 'max' in mode:
            if adaptive or mode == 'adaptive_max':
                return nn.AdaptiveMaxPool3d(**kwargs)
            else:
                return nn.MaxPool3d(**kwargs)
        elif 'avg' in mode:
            if adaptive or mode == 'adaptive_avg':
                return nn.AdaptiveAvgPool3d(**kwargs)
            else:
                return nn.AvgPool3d(**kwargs)
        else:
            raise ValueError(f'Only "max" or "avg" supported, not {mode}')
    else:
        raise ValueError(f'Only 2d or 3d supported, not {dim}')


class WeightInitializer:
    def __init__(self, init_mode=None, relu='relu'):
        """
        Initialize weights of network
        Slightly modified version of medical detection toolkit
        https://github.com/pfjaeger/medicaldetectiontoolkit/blob/master/utils/model_utils.py

        Parameters
        ----------
        init_mode : str
            Default None -> 'kaiming_uniform'
            specify the initialization method. 'xavier_uniform'|'xavier_normal'
            |'kaiming_uniform'|'kaiming_normal' | None
        relu : str
            string which specifies non-linearity (needed for kaiming init).
            Default: 'relu'

        See Also
        --------
        torch.nn.xavier_uniform_
        torch.nn.xavier_normal_
        torch.nn.kaiming_uniform_
        torch.nn.kaiming_normal_
        """
        self.init_mode = init_mode if init_mode is not None else \
            'kaiming_uniform'
        self._layers = [nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d,
                        nn.ConvTranspose3d, nn.Linear]
        self.relu = relu

    def __call__(self, net: torch.nn.Module, seed=0):
        """
        Initialize all weights of net

        Parameters
        ----------
        net : torch.nn.Module
            network which should be initialized
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        modules = [module for module in net.modules()
                   if type(module) in self._layers]
        for m in modules:
            if self.init_mode == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif self.init_mode == 'xavier_normal':
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif self.init_mode == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight.data,
                                         mode='fan_out',
                                         nonlinearity=self.relu,
                                         a=0)
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / np.sqrt(fan_out)
                    nn.init.uniform_(m.bias, -bound, bound)

            elif self.init_mode == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight.data,
                                        mode='fan_out',
                                        nonlinearity=self.relu,
                                        a=0)
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / np.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
