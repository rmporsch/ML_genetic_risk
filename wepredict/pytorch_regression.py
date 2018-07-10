"""Class and function for penalized regressions with tensorflow."""
import os
import numpy as np
import pickle
import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.preprocessing import scale


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
            penalty = F.sigmoid(self.loc-self.temp*self.gamma_zeta_ratio).sum()
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
        # print(out)
        # m = nn.Sigmoid()
        # out = m(out)
        return out, penalty


class _L12Norm(nn.Module):
    """L1 or L2 norm linear model."""

    def __init__(self, origin):
        """L1 or L2 norm linear model."""
        super(_L12Norm, self).__init__()
        self._origin = origin

    def _l1_reg(self):
        if self.training:
            penalty = Variable(torch.FloatTensor(1), requires_grad=True)
            penalty = torch.sum(torch.abs(self._origin.weight))
        else:
            penalty = 0
        return penalty

    def _l2_reg(self):
        if self.training:
            penalty = Variable(torch.FloatTensor(1), requires_grad=True)
            penalty = torch.sum(torch.mul(self._origin.weight,
                                          self._origin.weight))
        else:
            penalty = 0
        return penalty


class L12Linear(_L12Norm):
    """Linear model with L0 norm."""

    def __init__(self, in_features, out_features, penal, bias=True, **kwargs):
        """Linear model with L0 norm."""
        super(L12Linear, self).__init__(nn.Linear(in_features,
                                                  out_features,
                                                  bias=bias), **kwargs)
        self.penal = penal

    def forward(self, input):
        """Forward function with mask and penalty."""
        if self.penal == 'l1':
            penalty = self._l1_reg()
            out = F.linear(input, self._origin.weight, self._origin.bias)
        elif self.penal == 'l2':
            penalty = self._l2_reg()
            out = F.linear(input, self._origin.weight, self._origin.bias)
        else:
            raise ValueError('wrong norm specified')
        return out, penalty


class pytorch_linear(object):
    """Penalized regresssion with L1/L2/L0 norm."""

    def __init__(self, X, y, X_valid, y_valid, overwrite=False,
                 type='b', mini_batch_size=5000, if_shuffle=False):
        """Penalized regression."""
        super(pytorch_linear, self).__init__()
        self.n, self.input_dim = X.shape
        self.overwrite = overwrite
        self._shuffle_ids = np.arange(self.n)
        np.random.shuffle(self._shuffle_ids)
        self.output_dim = 1
        self.type = type
        self.mini_batch_size = mini_batch_size
        if if_shuffle:
            self.X = X[self._shuffle_ids, :]
            self.y = y[self._shuffle_ids].reshape(self.n, 1)
        else:
            self.X = X
            self.y = y.reshape(self.n, 1)
            # self.y = y
        self.X_valid = scale(X_valid.astype(np.float32))
        self.y_valid = scale(y_valid)

    def _model_builder(self, penal, **kwargs):
        if penal == 'l1':
            model = L12Linear(self.input_dim, self.output_dim, penal=penal)
        elif penal == 'l2':
            model = L12Linear(self.input_dim, self.output_dim, penal=penal)
        elif penal == 'l0':
            model = L0Linear(self.input_dim, self.output_dim, **kwargs)
        else:
            raise ValueError('incorrect norm specified')
        return model

    def _loss_function(self, labels, outputs):
        if self.type == 'c':
            loss = ((labels - outputs)**2).mean()
        elif self.type == 'b':
            loss = -(labels * torch.log(outputs)
                     + (1-labels)*torch.log(1-outputs)).mean()
        else:
            raise ValueError('wrong type specificed, either c or b')
        return loss

    def _accuracy(self, predict):
        if self.type == 'c':
            predict = predict.data.numpy().flatten()
            y_valid = self.y_valid.flatten()
            assert np.sum(~np.isfinite(predict)) == 0
            accu = np.corrcoef(y_valid, predict)
            accu = accu[0][1]
            accu = accu**2
        else:
            accu = np.mean(np.round(predict.data.numpy()) == self.y_valid)
        return accu

    def _iterator(self, x, y):
        start = 0
        end = self.mini_batch_size
        index = np.arange(self.n)
        while True:
            if end >= self.n:
                end = end - self.n
            if start >= self.n:
                start = start - self.n
            bool_index = ~((index >= start) ^ (index < end))
            if start > end:
                bool_index = ~(bool_index)
            assert np.sum(bool_index) == self.mini_batch_size
            xyield = scale(x[bool_index, :].astype(np.float32))
            yyield = scale(y[bool_index, :])
            # yyield = scale(y[bool_index])
            yield xyield, yyield
            start = np.abs(end)
            end = start + self.mini_batch_size

    def run(self, penal: str = 'l1', lamb: float = 0.01,
            epochs: int = 201, l_rate: float = 0.01, **kwargs):
        """Run regression with the given paramters."""
        model = self._model_builder(penal, **kwargs)
        dataset = self._iterator(self.X.astype(np.int32), self.y)
        optimizer = torch.optim.SGD(model.parameters(), lr=l_rate)
        valid_x = Variable(torch.from_numpy(self.X_valid)).float()
        save_loss = list()
        save_pred = list()
        for _ in range(epochs):
            xx, yy = next(dataset)
            # xx, yy = self.X, self.y
            input = Variable(torch.from_numpy(xx)).float()
            labels = Variable(torch.from_numpy(yy)).float()
            optimizer.zero_grad()
            outputs, penalty = model.forward(input)
            loss = self._loss_function(labels, outputs)
            loss = loss + lamb*penalty
            loss.backward()
            optimizer.step()
            if (_+1) % 100 == 0:
                # print("epoch {}, loss {}, norm {}".format(_, loss.item(),
                #                                           penalty.item()))
                predict, penalty = model.forward(valid_x)
                accu = self._accuracy(predict)
                save_pred.append(accu)
                # if np.allclose(save_loss[-10], loss.item(), 1e-4):
                #     break
            save_loss.append(loss.item())
        predict, penalty = model.forward(valid_x)
        accu = self._accuracy(predict)
        coef = []
        for name, param in model.named_parameters():
            coef.append(param.data.numpy())
        param = {}
        prediction = predict.data.numpy().flatten()
        param['lambda'] = lamb
        param['epoch'] = epochs
        param['penal'] = penal
        param['type'] = self.type
        param['loss'] = save_loss
        param['pred'] = save_pred
        return {'accu': accu, 'param': param,
                'pred': prediction, 'coef': coef}
