import pickle


def load_pickle(path: str):
    """
    Load a single pickled sample

    Parameters
    ----------
    path : str
        path to sample

    Returns
    -------
    object
        loaded data
    """
    with open(path, "rb") as f:
        return pickle.load(f)
