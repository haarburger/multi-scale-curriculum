import numpy as np
import typing
import pickle
import os


def create_single_sample(size_patch: typing.Iterable, size_obj: tuple,
                         otype: str, sid: int, path: str, offset=0.2):
    """
    Generate a single sample

    Parameters
    ----------
    size_patch : typing.Iterable
        size of a single sample
    size_obj : typing.Iterable
        size of the onbject inside the sample
    otype : str
        object type. Supported are "square" and "circle".
    id: int
        sample id
    path: str
        path to folder where samples should be saved

    Raises
    ------
    ValueError
        Is raised if otype is not supported.
    """
    print("Processing id {} of type {}".format(sid, otype))
    patch = np.random.rand(*size_patch).astype(np.float)
    mask = np.zeros_like(patch).astype(np.uint8)

    centre = np.asarray([np.random.randint(size_obj[i] // 2,
                                           sp - size_obj[i] // 2)
                         for i, sp in enumerate(size_patch)])
    size_obj = np.asarray(size_obj)

    if otype == "circle":
        with np.nditer(patch, flags=['multi_index']) as it:
            while not it.finished:
                dist = centre - np.asarray(it.multi_index)
                if all(dist < size_obj):
                    mask[it.multi_index] = 1
                it.iternext()
        label = 0
    elif otype == "square":
        slicing = tuple([slice(c - size_obj[i] // 2, c + size_obj[i] // 2)
                         for i, c in enumerate(centre)])
        mask[slicing] = 1
        label = 1
    else:
        raise ValueError("{} is not a supported otype.".format(otype))

    patch[mask.astype(np.bool)] += offset
    patch = np.clip(patch, 0, 1)
    data = {'data': patch[None], 'mask': mask[None], 'label': label, 'id': sid}
    with open(os.path.join(path, "{}.pkl".format(sid)), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    size_patch = (16, 128, 128)  # size of single sample
    size_obj = (3, 20, 20)  # size of object inside sample
    basepath = "/tmp/mscl"  # path to dir
    num_samples = {"train": 200, "val": 100, "test": 50}
    os.mkdir(basepath)
    for key, item in num_samples.items():
        path = os.path.join(basepath, str(key))
        if not os.path.isdir(path):
            os.mkdir(path)
        otype = np.random.choice(["square", "circle"], item)

        calls = [(size_patch, size_obj, i, idx, path)
                 for idx, i in enumerate(otype)]

        for c in calls:
            create_single_sample(*c)
