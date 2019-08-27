import torch.nn as nn
import torch.nn.functional as F
import torch
from mscl.utils import ConfigHandlerAbstract, NDConvGenerator


class ResBlock(nn.Module):
    def __init__(self, start_filts, planes, conv, stride=1, downsample=None,
                 norm=None, relu='relu', bias=True):
        """
        ResBlock with bottleneck

        Parameters
        ----------
        start_filts : int
            number of input channels
        planes : int
            number of intermediate channels
        conv : NDConvGenerator
            wrapper for 2d/3d conv layer with norm and relu
        stride : int
            stride for the first convolution
        downsample : tuple
            first entry defines input channels into down-sampling conv,
            second is the multiplicative factor for the output channels,
            third is the stride for the down-sampling convolution
        norm : str
            defines the norm which should be used.
            See :class: `NDConvGenerator` for more info
        relu : str
            defines the non linearity which should be used.
            See :class: `NDConvGenerator` for more info
        bias : bool
            disabled bias for convolutions

        See Also
        --------
        :class: `NDConvGenerator`
        """
        super(ResBlock, self).__init__()
        self.conv1 = conv(start_filts, planes, ks=1, stride=stride,
                          norm=norm, relu=relu, bias=bias)
        self.conv2 = conv(planes, planes, ks=3, pad=1,
                          norm=norm, relu=relu, bias=bias)
        self.conv3 = conv(planes, planes * 4, ks=1,
                          norm=norm, relu=None, bias=bias)
        self.relu = nn.ReLU(inplace=True) if relu == 'relu' else \
            nn.LeakyReLU(inplace=True)
        if downsample is not None:
            self.downsample = \
                conv(downsample[0], downsample[0] * downsample[1], ks=1,
                     stride=downsample[2], norm=norm, relu=None, bias=bias)
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResBlockPlain(nn.Module):
    def __init__(self, start_filts, planes, conv, stride=1, downsample=None,
                 norm=None, relu='relu', bias=True):
        """
        ResBlock

        Parameters
        ----------
        start_filts : int
            number of input channels
        planes : int
            number of intermediate channels
        conv : NDConvGenerator
            wrapper for 2d/3d conv layer with norm and relu
        stride : int
            stride for the first convolution
        downsample : tuple
            first entry defines input channels into down-sampling conv,
            second is the multiplicative factor for the output channels,
            third is the stride for the down-sampling convolution
        norm : str
            defines the norm which should be used.
            See :class: `NDConvGenerator` for more info
        relu : str
            defines the non linearity which should be used.
            See :class: `NDConvGenerator` for more info
        bias : bool
            disabled bias for convolutions

        See Also
        --------
        :class: `NDConvGenerator`
        """
        super(ResBlockPlain, self).__init__()
        self.conv1 = conv(start_filts, planes, ks=3, pad=1, stride=stride,
                          norm=norm, relu=relu, bias=bias)
        self.conv2 = conv(planes, planes, ks=3, pad=1,
                          norm=norm, relu=None, bias=bias)
        self.relu = nn.ReLU(inplace=True) if relu == 'relu' else \
            nn.LeakyReLU(inplace=True)
        if downsample is not None:
            self.downsample = \
                conv(downsample[0], downsample[0] * downsample[1], ks=1,
                     stride=downsample[2], norm=norm, relu=None, bias=bias)
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# Define the settings for different ResNet architectures
RESNETS = {
    'resnet10': {'block_list': [1, 1, 1, 1], 'block': ResBlockPlain, 'expansion': 1},
    'resnet18': {'block_list': [2, 2, 2, 2], 'block': ResBlockPlain, 'expansion': 1},
    'resnet34': {'block_list': [3, 4, 6, 3], 'block': ResBlockPlain, 'expansion': 1},
    'resnet24': {'block_list': [2, 2, 2, 2], 'block': ResBlock, 'expansion': 4},
    'resnet50': {'block_list': [3, 4, 6, 3], 'block': ResBlock, 'expansion': 4},
    'resnet101': {'block_list': [3, 4, 23, 3], 'block': ResBlock, 'expansion': 4},
    'resnet151': {'block_list': [3, 8, 36, 3], 'block': ResBlock, 'expansion': 4}
}


class ResNetBackbone(nn.Module):
    def __init__(self, ch: ConfigHandlerAbstract, conv: NDConvGenerator):
        """
        Build ResNet model

        Parameters
        ----------
        ch : ConfigHandlerAbstract
            config handler containing all settings
            Required keys:
                architecture : str
                    specifies the architecture. Supported: resnet18|resnet32|
                    resnet24|resnet50|resnet101|resnet152
                start_filts : int
                    number of channels after first convolution
                operate_stride1 : bool
                    insert additional convolutions at top layer for segmentation
                in_channels : int
                    number of input channels
                norm : str
                    normalization for conv layer
                relu : str
                    non-linearity for conv layer
            Optional keys:
                reduced_pool : bool
                    reduces pooling in z-direction
        conv: NDConvGenerator
            wrapper for 2d/3d convolutions

        See Also
        --------
        :class: `NDConvGenerator`, :class: `ConfigHandlerAbstract`
        """
        super().__init__()
        # get settings for specific resnet
        self.n_blocks = RESNETS[ch['architecture']]['block_list']
        self.block = RESNETS[ch['architecture']]['block']
        self.block_expansion = RESNETS[ch['architecture']]['expansion']

        # adjust resnet
        start_filts = ch['start_filts']
        self.operate_stride1 = False  # not needed to reproduce results
        self.reduced_pool = ch['reduced_pool'] if 'reduced_pool' in ch else \
            False
        self.dim = conv.dim

        in_channels = ch['in_channels']
        norm = ch['norm']
        relu = ch['relu']
        # disable bias when norm is used
        bias = True if norm is None else False

        if self.operate_stride1:
            self.C0 = nn.Sequential(
                conv(in_channels, start_filts, ks=3, pad=1,
                     norm=norm, relu=relu, bias=bias),
                conv(start_filts, start_filts, ks=3, pad=1,
                     norm=norm, relu=relu, bias=bias))

            self.C1 = conv(start_filts, start_filts, ks=7,
                           stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3,
                           norm=norm, relu=relu, bias=bias)

        else:
            self.C1 = conv(in_channels, start_filts, ks=7,
                           stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3,
                           norm=norm, relu=relu, bias=bias)

        start_filts_exp = start_filts * self.block_expansion

        C2_layers = []
        C2_layers.append(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if conv.dim == 2
            else nn.MaxPool3d(kernel_size=3, stride=(2, 2, 1), padding=1))
        C2_layers.append(
            self.block(start_filts, start_filts, conv=conv, stride=1,
                       norm=norm, relu=relu, bias=bias,
                       downsample=(start_filts, self.block_expansion, 1)))
        for i in range(1, self.n_blocks[0]):
            C2_layers.append(
                self.block(start_filts_exp, start_filts, conv=conv,
                           norm=norm, relu=relu, bias=bias))
        self.C2 = nn.Sequential(*C2_layers)

        C3_layers = []
        if self.reduced_pool:
            C3_layers.append(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if
                conv.dim == 2 else nn.MaxPool3d(kernel_size=3,
                                                stride=(2, 2, 1), padding=1))
            C3_layers.append(
                self.block(start_filts_exp, start_filts * 2, conv=conv,
                           stride=1, norm=norm, relu=relu, bias=bias,
                           downsample=(start_filts_exp, 2, 1)))
        else:
            C3_layers.append(
                self.block(start_filts_exp, start_filts * 2, conv=conv,
                           stride=2, norm=norm, relu=relu, bias=bias,
                           downsample=(start_filts_exp, 2, 2)))
        for i in range(1, self.n_blocks[1]):
            C3_layers.append(
                self.block(start_filts_exp * 2, start_filts * 2, conv=conv,
                           norm=norm, relu=relu, bias=bias))
        self.C3 = nn.Sequential(*C3_layers)

        C4_layers = []
        C4_layers.append(self.block(start_filts_exp * 2, start_filts * 4,
                                    conv=conv, stride=2, norm=norm,
                                    relu=relu, bias=bias,
                                    downsample=(start_filts_exp * 2, 2, 2)))
        for i in range(1, self.n_blocks[2]):
            C4_layers.append(
                self.block(start_filts_exp * 4, start_filts * 4, conv=conv,
                           norm=norm, relu=relu, bias=bias))
        self.C4 = nn.Sequential(*C4_layers)

        C5_layers = []
        C5_layers.append(
            self.block(start_filts_exp * 4, start_filts * 8, conv=conv,
                       stride=2, norm=norm, relu=relu, bias=bias,
                       downsample=(start_filts_exp * 4, 2, 2)))
        for i in range(1, self.n_blocks[3]):
            C5_layers.append(
                self.block(start_filts_exp * 8, start_filts * 8, conv=conv,
                           norm=norm, relu=relu, bias=bias))
        self.C5 = nn.Sequential(*C5_layers)

    def forward(self, x):
        """
        Forward input through network

        Parameters
        ----------
        x : torch.Tensor
            image tensor with shape (N,C,Y,X,Z)

        Returns
        -------
        list of output feature maps
        """
        if self.operate_stride1:
            c0_out = self.C0(x)
        else:
            c0_out = x

        c1_out = self.C1(c0_out)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        c5_out = self.C5(c4_out)

        out_list = [c1_out, c2_out, c3_out, c4_out, c5_out]

        if self.operate_stride1:
            out_list = [c0_out] + out_list
        return out_list
