import torch
import torch.nn as nn

#TODO: remove
class LogNllLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Compute cross entropy loss after softmax was applied to output

        Parameters
        ----------
        args :
            variable number positional arguments passed to NLLLoss
        kwargs :
            variable number of keyword arguments passed to NLLLoss
        """
        super().__init__()
        self._nll = nn.NLLLoss(*args, **kwargs)

    def forward(self, input, target):
        """
        Compute loss

        Parameters
        ----------
        input : torch.tensor
            input tensor

        Returns
        -------
        torch.tensor
            computed loss
        """
        return self._nll(torch.log(input), target)
