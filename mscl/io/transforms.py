from batchgenerators.transforms import AbstractTransform
import numpy as np


class GenerateOhe(AbstractTransform):
    def __init__(self, num_classes, key_in, key_out=None):
        """
        Generates one hot encoding of segmentation

        Parameters
        ----------
        num_classes: int
            number of classes
        key_in: list of hashable
            key of segmentation in input dict
        key_out: list of hashable
            Maps the inputs keys to output keys. None creates a new key with
            an _ohe extension at the end. Default: None
            Note: if key already exists the data inside the key is overwritten
        """
        super().__init__()
        self.num_classes = num_classes
        self.key_in = key_in
        self.key_out = key_out

    def __call__(self, **data_dict):
        """
        Performs transformation

        Parameters
        ----------
        data_dict: dict
            dict with data

        Returns
        -------
        dict
            dict with additional key

        Raises
        ------
        ValueError
            if key_in and key_out are not the same length (except if
            key_out is None)
        KeyError
            if key_in does not exist in data_dict
        """
        if self.key_out is None:
            self.key_out = [str(k) + '_ohe' for k in self.key_in]

        if not len(self.key_in) == len(self.key_out):
            raise ValueError('Key_out needs be the same length as as Key_in')

        # iterate over keys
        for i, ik in enumerate(self.key_in):
            if ik not in data_dict:
                raise KeyError(f'Key in \'{ik}\' not in data dict')

            data_dict[self.key_out[i]] = self.encode_batch_ohe(
                self.num_classes, data_dict[ik])

        return data_dict


class PopKeys(AbstractTransform):
    def __init__(self, *args):
        """
        Pop keys from batch which are no longer needed
        Parameters
        ----------
        args: strings
            keys for data_dict
        """
        super().__init__()
        self.args = args

    def __call__(self, **data_dict):
        """Pop keys

        Returns
        -------
        dict
            new data dict
        """
        for arg in self.args:
            data_dict.pop(arg, None)
        return data_dict
