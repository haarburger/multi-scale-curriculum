import logging
import numpy as np

from functools import reduce
from skimage.measure import regionprops
from batchgenerators.augmentations.utils import pad_nd_image

logger = logging.getLogger(__name__)


class LoadPatches:
    def __init__(self, load_fn, patch_size, mask_key='mask',
                 mapping_key=None, crop_keys=('data',), pop_keys=tuple(),
                 min_dist=None, discard_mul_fg=False, random_offset=False,
                 skip_no_background=False, **kwargs):
        """
        Loading function for patches
        default: extracts patches from mask and data
        default: mask is mapped to respective class for segmentation

        Parameters
        ----------
        load_fn : function
            function to load a single sample. Needs to provide mask_key. Mask
            is used to crop patches around foreground classes.
        patch_size : tuple
            size of patch
        mask_key : hashable
            load needs to provide the mask in this key. The mask in channel
            zero is used for locating the foreground classes.
        mapping_key : hashable
            if None this does nothing. if provided a label is generated
            where the label is determined by the mapping from mask values to
            classes (the maximum of all present classes is selected as the
            final label). Result is saved in `label`
        crop_keys : list of str
            additional keys where patches should be extracted from
        pop_keys : list of str
            keys which should be removed (not forwarded to patches)
        min_dist : numpy.ndarray
            minimum distance of batch border to sample border.
        discard_mul_fg : bool
            discard patch which contains more than one foreground object
        random_offset : bool
            disabled random offset of lesion in patches
        skip_no_background : bool
            skip patches which do not contain any background
        kwargs :
            variable number of keyword arguments passed to load_sample function
        """
        self.load_fn = load_fn

        # save settings
        self.patch_size = patch_size
        self.mask_key = mask_key
        self.mapping_key = mapping_key

        self.min_dist = min_dist
        self.discard_mul_fg = discard_mul_fg
        self.rand_offset = random_offset
        self.skip_no_background = skip_no_background

        self.crop_keys = crop_keys
        self.pop_keys = pop_keys
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        """
        Load Sample and extract patches

        Parameters
        ----------
        args :
            passed to self.load_sample
        kwargs :
            passed to self.load_sample

        Returns
        -------
        list of dict
            contains data
        """
        patch_list = []

        # load single sample
        sample = self.load_fn(*args, **kwargs, **self.kwargs)

        # pop keys which should be cropped
        data_dict = {self.mask_key: sample.pop(self.mask_key)}
        for key in self.crop_keys:
            data_dict[key] = sample.pop(key)

        # pop keys which shound not be forwarded
        _ = [sample.pop(key, None) for key in self.pop_keys]

        # sample patches around lesion
        patches = self.extract_patch(data_dict, sample)
        patch_list += patches
        return patch_list

    def extract_patch(self, data_dict, sample):
        """
        Samples patches around foreground objects

        Parameters
        ----------
        data_dict : dict
            dict with data which should be cropped
        sample : dict
           additional information added to every sample

        Returns
        -------
        list of dict
            patches
        """
        patch_list = []
        # only sample if at least one lesion is present
        if data_dict[self.mask_key].max() > 0:
            centre = [i['centroid'] for i in regionprops(
                data_dict[self.mask_key][0])]

            for c in centre:
                random_offset = np.divide(self.patch_size, 4) if \
                    self.rand_offset else None

                # extract coordinates
                slice_list = LoadPatches.get_patch_coordinates(
                    c, self.patch_size, data_dict[self.mask_key].shape[1:],
                    random_offset=random_offset,
                    min_dist=self.min_dist)

                # check if extraction was successful
                if slice_list is None:
                    continue

                # extract patches and pad data to patch size
                patch_dict = {}
                for key, item in data_dict.items():
                    # use all channels
                    slices = [slice(0, item.shape[0])] + slice_list
                    patch = np.copy(item[tuple(slices)])
                    if np.any(patch.shape[1:] < tuple(self.patch_size)):
                        patch = pad_nd_image(patch, self.patch_size,
                                             mode='constant',
                                             kwargs={'constant_values': 0})
                    patch_dict[key] = patch

                # skip patches without background
                if self.skip_no_background:
                    if not (patch_dict[self.mask_key] == 0).any():
                        continue

                # skip samples with multiple foreground objects (>2, 1 bg, 1 fg)
                if self.discard_mul_fg and \
                        np.unique(patch_dict[self.mask_key]).shape[0] > 2:
                    continue

                # generate label for patch
                if self.mapping_key is not None:
                    patch_values = np.unique(patch_dict[self.mask_key])
                    patch_values = patch_values[patch_values > 0]
                    mapping = sample[self.mapping_key]
                    patch_classes = [mapping[mapping[:, 0] == i][0, 1]
                                     for i in patch_values
                                     if (mapping[:, 0] == i).any()]
                    if len(patch_classes) > 0:
                        patch_dict['label'] = np.array(np.max(patch_classes))
                    else:
                        if 'id' not in sample:
                            logger.warning(f'Skip: no patch classes found.')
                        else:
                            logger.warning(f'Skip: {sample["id"]} no patch classes found.')
                        print("Mapping: {}".format(mapping))
                        print("Val: {}".format(patch_values))
                        continue

                patch_list.append({**patch_dict, **sample})
        return patch_list

    @staticmethod
    def get_patch_coordinates(centre, patch_size, data_shape,
                              random_offset=None, min_dist=None):
        """
        Generates list with slices to crop patches from data

        Parameters
        ----------
        centre : :class: 'numpy.ndarray'
            coordinates of centre
        patch_size : iterable
            patch size which should be extracted
        data_shape : :class: 'numpy.ndarray'
            original data shape
        random_offset : iterable
            range for a random offset for the centre coordinates
        min_dist : :class: 'numpy.ndarray'
            minimum distance of patch border from data border

        Returns
        -------
        list
            list with slices to index original array
        """
        # generate random offset
        offset = 0
        if random_offset is not None:
            offset = np.asarray([np.random.randint(-i, i + 1)
                                 for i in random_offset])
            centre += offset

        # clip data
        lb = np.maximum(centre - np.array(patch_size) // 2, 0).astype(np.int64)
        ub = np.minimum(
            centre + np.array(patch_size) // 2, data_shape).astype(np.int64)

        # check for minimum distance to border
        if min_dist is not None:
            if np.any(lb < min_dist) or np.any(data_shape-ub < min_dist):
                if np.all(offset == 0):
                    # not possible to crop from original centre -> skip sample
                    return None
                else:
                    # remove random offset and try to crop from original centre
                    return LoadPatches.get_patch_coordinates(
                        centre - offset,
                        patch_size,
                        data_shape,
                        random_offset=None,
                        min_dist=min_dist)

        slice_list = [slice(l, u) for l, u in zip(lb, ub)]
        return slice_list


class LoadPatchesBackground(LoadPatches):
    def __init__(self, fseg_area, prob, *args,
                 pop_fseg=True, fseg_key='fseg', max_iter=100, **kwargs):
        """
        Samples additional background samples for class 0
        Parameters
        ----------
        fseg_area : float
            percentage of patch which contains segmentation
        prob : float or int
            if <1 this is interpredted as an probability to extract one
            background patch from a sample. if >=1 this is interpreted
            as the number of patches which should be extracted from each
            sample
        pop_fseg : bool
            pop segmentation from samples after sampling from background
        fseg_key : hashable
            key for segmentation
        max_iter : int
            maximum number of iterations
        args :
            variable number of positional arguments passed to super class
        kwargs :
            variable number of keyword arguments pass to super class

        See Also
        --------
        :class: `LoadPatches`
        """
        super().__init__(*args, **kwargs)
        self.fseg_area = fseg_area
        self.pop_fseg = pop_fseg
        self.fseg_key = fseg_key
        self.max_iter = max_iter

        if prob < 1:
            self._num = 1
            self._prob = prob
        else:
            self._num = prob
            self._prob = 1

    def __call__(self, *args, **kwargs):
        """
        Load Sample and extract patches

        Parameters
        ----------
        args :
            passed to self.load_sample
        kwargs :
            passed to self.load_sample

        Returns
        -------
        list of dict
            contains data
        """
        patch_list = []

        # load single sample
        sample = self.load_fn(*args, **kwargs, **self.kwargs)

        # pop keys which should be cropped
        data_dict = {self.mask_key: sample.pop(self.mask_key)}
        for key in self.crop_keys:
            data_dict[key] = sample.pop(key)

        # pop keys which shound not be forwarded
        _ = [sample.pop(key, None) for key in self.pop_keys]

        # sample patches around lesion
        patch_list += self.extract_patch(data_dict, sample)
        patch_list += self.extract_background(data_dict, sample)
        return patch_list

    def extract_background(self, data_dict, sample):
        """
        Samples background samples with regard to segmentation

        Parameters
        ----------
        data_dict : dict
            dict with data which should be cropped
        sample : dict
            additional information which should be passed through to all patches
            Needs to provide the segmentation information!

        Returns
        -------
        list of dict
            patches
        """
        r = np.random.rand(1)
        # only generate patches with probability
        if not (r < self._prob):
            return []

        patch_list = []
        fseg = sample.pop(self.fseg_key)
        fseg[fseg > 0] = 1  # unify all classes

        # sample multiple patches
        for _ in range(self._num):
            iter_count = 0  # count iterations
            found_patch = False
            # iterate until patch found or max_iter
            while not found_patch:
                # check for iterations
                iter_count += 1
                if iter_count >= self.max_iter:
                    break

                slice_list = None
                # if it was not possible to extract patch
                while slice_list is None:
                    centre = np.asarray([np.random.randint(0, i)
                                         for i in data_dict['data'].shape[1:]])

                    # extract coordinates
                    slice_list = LoadPatches.get_patch_coordinates(
                        centre, self.patch_size, data_dict['data'].shape[1:],
                        random_offset=np.divide(self.patch_size, 4),
                        min_dist=self.min_dist)

                # skip patch if it does not contain enough foreground
                slices = [slice(0, fseg.shape[0])] + slice_list
                patch_fseg = np.copy(fseg[tuple(slices)])
                patcharea = reduce(lambda x, y: x*y, patch_fseg.shape[1:])
                if (patch_fseg.sum()/patcharea) < self.fseg_area:
                    continue

                # extract patches and pad data to patch size
                patch_dict = {}
                for key, item in data_dict.items():
                    slices = [slice(0, item.shape[0])] + slice_list
                    patch = np.copy(item[tuple(slices)])
                    if np.any(patch.shape[1:] < tuple(self.patch_size)):
                        patch = pad_nd_image(patch, self.patch_size,
                                             mode='constant',
                                             kwargs={'constant_values': 0})
                    patch_dict[key] = patch

                # discard patches which contain lesions
                if np.max(patch_dict[self.mask_key]) > 0:
                    continue

                # generate label
                if self.mapping_key is not None:
                    patch_dict['label'] = np.array(0)  # patch label

                if not self.pop_fseg:
                    patch_dict[self.fseg_key] = fseg

                found_patch = True
                patch_list.append({**patch_dict, **sample})
        return patch_list
