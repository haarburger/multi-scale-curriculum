import torch


def metric_wrapper_pytorch(fct, device=torch.device('cpu'),
                           prep_batch=None, **fn_kwargs):
    """
    Provide a wrapper to compute metrics with pytorch functions

    Parameters
    ----------
    fct : callable
        pytorch function to compute metric
    prep_batch : callable
        prepare inputs for metric, e.g. cast to needed type
        (Needs to return a list and dict)

    Returns
    -------
    float
        result
    """
    def pytorch_metric(*args, **kwargs):
        args_pytorch = [torch.from_numpy(arg).to(
            device=device) for arg in args]
        kwargs_pytorch = {k: torch.from_numpy(v).to(
            device=device) for k, v in kwargs.items()}
        if prep_batch is not None:
            args_pytorch, kwargs_pytorch = prep_batch(
                *args_pytorch, **kwargs_pytorch)
        return fct(*args_pytorch, **kwargs_pytorch,
                   **fn_kwargs).detach().cpu().numpy()
    return pytorch_metric


def prep_classification(*args, **kwargs):
    """
    Cast arguments to correct type for classification metrics
    """
    # calc metric only uses positional arguments
    args = list(args)
    # first argument is the prediction
    args[0] = args[0].float()
    # second argument is the label
    args[1] = args[1].long()
    return args, kwargs
