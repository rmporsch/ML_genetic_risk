"""L0 Norm."""
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.autograd import Variable
import numpy as np

def hard_sigmoid(x):
    """Hard Sigmoid function."""
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))

class _L0Norm(nn.Module):
    """L0 norm."""

    def __init__(self, origin, loc_mean: float = 0.0,
                 loc_sdev: float = 0.01,
                 beta: float = 2/3, gamma: float = -0.1,
                 zeta: float = 1.1, fix_temp: bool = True):
        """Class of layers using L0 Norm.

        :param origin: original layer such as nn.Linear(..), nn.Conv2d(..)
        :param loc_mean: mean of the normal of initial location parameters
        :param loc_sdev: standard deviation of initial location parameters
        :param beta: initial temperature parameter
        :param gamma: lower bound of "stretched" s
        :param zeta: upper bound of "stretched" s
        :param fix_temp: True if temperature is fixed
        """
        super(_L0Norm, self).__init__()
        self._origin = origin
        self._size = self._origin.weight.size()
        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean,
                                                                loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(
            torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros(self._size))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = np.log(-gamma / zeta)

    def _get_mask(self):
        if self.training:
            self.uniform.uniform_()
            u = Variable(self.uniform)
            s = F.sigmoid((torch.log(u)-torch.log(1-u)+self.loc)/self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = F.sigmoid((self.loc-self.temp*self.gamma_zeta_ratio)*(self._origin.weight**2)).sum()
            # penalty = F.sigmoid(self.loc-self.temp*self.gamma_zeta_ratio).sum()
        else:
            s = F.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
            penalty = 0
        return hard_sigmoid(s), penalty


class L0Linear(_L0Norm):
    """Linear model with L0 norm."""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, **kwargs):
        """Linear model with L0 norm."""
        super(L0Linear, self).__init__(nn.Linear(in_features, out_features,
                                                 bias=bias), **kwargs)

    def forward(self, input):
        """Forward function with mask and penalty."""
        mask, penalty = self._get_mask()
        out = F.linear(input, self._origin.weight * mask, self._origin.bias)
        return out, penalty
