import numpy as np
import mscl.models.model


def create_scales_config(ch):
    """
    Computes the feature map sizes and backbone strides

    Parameters
    ----------
    ch: ConfigHandlerAbstract
        Required Keys:
            'dim': int
                determines dimensionality of convolution
            'path_size': list of int
                size of input image
            'backbone_stride': list of int
                stride inside backbone (depends on pooling)
    """
    back_cls = mscl.models.model.get_back(ch)
    strides = back_cls.compute_strides(ch)
    ch['backbone_strides'] = strides

    if ch['dim'] == 2:
        ch['backbone_shapes'] = np.array(
            [[int(np.ceil(ch['augment.patch_size'][0] / stride)),
              int(np.ceil(ch['augment.patch_size'][1] / stride))]
             for stride in ch['backbone_strides']['xy']])
    else:
        ch['backbone_shapes'] = np.array(
            [[int(np.ceil(ch['augment.patch_size'][0] / stride)),
              int(np.ceil(ch['augment.patch_size'][1] / stride)),
              int(np.ceil(ch['augment.patch_size'][2] / stride_z))]
             for stride, stride_z in zip(ch['backbone_strides']['xy'],
                                         ch['backbone_strides']['z'])])
    return ch


def create_filts_config(ch):
    """
    Generates the number of filters inside the network

    Parameters
    ----------
    ch: ConfigHandlerAbstract
        Required Keys:
            'architecture': str
                determines the architecture for the backbone
    """
    back_cls = mscl.models.model.get_back(ch)
    ch['filts'] = back_cls.compute_filts(ch)
    return ch
