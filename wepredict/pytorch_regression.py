"""Class and function for penalized regressions with tensorflow."""
import numpy as np
import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import scale
import logging
from wepredict.l0_norm import L0Linear
from memory_profiler import profile
from sklearn.utils import shuffle

lg = logging.getLogger(__name__)

def hard_sigmoid(x):
    """Hard Sigmoid function."""
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))


class ResultCollector(object):

    def __init__(self,n: int, p: int, lamb: float, epochs: int, type: str, penal_type: str):
        self.start_time = datetime.datetime.now()
        self.loss = list()
        self.iter_accu_valid = list()
        self.iter_accu_training = list()
        self.iter_coefs = list()
        self.regularization_value = lamb
        self.epoch = epochs
        self.penal = penal_type
        self.type = type
        self.n = n
        self.p = p

        self.accu = None
        self.coef = None
        self.prediction_valid = None
        self.end_time = None

    def final_results(self, accu, prediction_valid, coef):
        self.accu = accu
        self.coef = coef
        self.prediction_valid = prediction_valid
        self.end_time = datetime.datetime.now()
        lg.info('The coef field has the following names %s',
                self.coef.keys())

    def add_valid(self, pred_valid):
        self.iter_accu_valid.append(pred_valid)

    def add_parameters(self, coefs):
        """
        Append parameters to result class.

        :param coefs: paramter class from PyTorch
        :return: None
        """
        out = {}
        for name, param in coefs:
            new_name = name.replace('_origin.', '')
            out[new_name] = param.data.numpy()
        self.iter_coefs.append(out)


class RegL0(nn.Module):
    """
    Run Regression with L0
    """

    def __init__(self, n_input, n_output, mean: float = 1.0, **kwargs):
        super(RegL0, self).__init__()
        self.linear = L0Linear(n_input, n_output, loc_mean=mean, **kwargs)

    def forward(self, x, training = True):
        self.linear.training = training
        x, z1 = self.linear(x)
        self.linear.training = False
        return x, z1

class RegL1(nn.Module):
    """
    Run Regression with L1
    """

    def __init__(self, n_input, n_output):
        super(RegL1, self).__init__()
        self.linear = nn.Linear(n_input, n_output, bias=True)

    def forward(self, x, training = True):
        self.training = training
        x = self.linear(x)
        z1 = torch.sum(torch.abs(self.linear.weight))
        self.training = True
        return x, z1


class pytorch_linear(object):
    """Penalized regresssion with L1/L2/L0 norm."""

    def __init__(self, X, y, X_valid, y_valid,
                 type='b', mini_batch_size=5000, if_shuffle=False):
        """Penalized regression."""
        super(pytorch_linear, self).__init__()
        self.n, self.input_dim = X.shape
        self._shuffle_ids = np.arange(self.n)
        self.output_dim = 1
        self.type = type
        self.mini_batch_size = mini_batch_size
        if if_shuffle:
            self.X = X[self._shuffle_ids, :]
            self.y = y[self._shuffle_ids].reshape(self.n, -0.5)
        else:
            self.X = X
            self.y = y.reshape(self.n, 1)
            # self.y = y
        self.X_valid = scale(X_valid.astype(np.float32))
        self.y_valid = scale(y_valid)

    def _model_builder(self, penal, **kwargs):
        if penal == 'l1':
            model = RegL1(self.input_dim, self.output_dim)
        elif penal == 'l0':
            model = RegL0(self.input_dim, self.output_dim)
        elif penal == 'l02':
            model = RegL0(self.input_dim, self.output_dim, l02=True)
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
            if not (np.sum(~np.isfinite(predict)) == 0):
                return 0.0
            accu = np.corrcoef(y_valid, predict)
            accu = accu[0][1]
        else:
            accu = np.mean(np.round(predict.data.numpy()) == self.y_valid)
        return accu

    def iterator(self):
        """
        Iterator over x,y paramters with given batch size.

        :return: None
        """
        start = 0
        end = self.mini_batch_size
        index = np.arange(self.n)
        self.saver_check = []
        iter_over_all = False
        while True:
            if end >= self.n:
                end = end - self.n
            if start >= self.n:
                start = start - self.n
            bool_index = ~((index >= start) ^ (index < end))
            if start > end:
                bool_index = ~(bool_index)
                iter_over_all = True
            assert np.sum(bool_index) == self.mini_batch_size
            self.saver_check.append(bool_index)
            xyield = np.zeros((self.mini_batch_size,
                               self.input_dim),
                              dtype=float)
            batch_index = index[bool_index]
            yyield = np.zeros((self.mini_batch_size, 1))
            for u, i in np.ndenumerate(batch_index):
                xyield[u] = self.X[i]
                yyield[u] = self.y[i]
            xyield = scale(xyield)
            yyield = scale(yyield)
            start = np.abs(end)
            end = start + self.mini_batch_size
            yield xyield, yyield, iter_over_all

    def run(self, penal: str = 'l1', lamb: float = 0.01,
            epochs: int = 201, l_rate: float = 0.01, logging_freq=100):
        """
        Run penalized regression.

        :param penal: penalty either l0,l1, or l2
        :param lamb: regularization parameter
        :param epochs: number of epochs
        :param l_rate: learning rate
        :param logging_freq: logging frequency
        :return:
        """
        results = ResultCollector(self.n, self.input_dim, lamb, epochs, self.type, penal)
        model = self._model_builder(penal)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=l_rate)
        valid_x = Variable(torch.from_numpy(self.X_valid)).float()
        for _ in range(epochs):
            self.X, self.y = shuffle(self.X, self.y)
            dataset = self.iterator()
            done_all = False
            while done_all is False:
                xx, yy, done_all = next(dataset)
                input = Variable(torch.from_numpy(xx)).float()
                labels = Variable(torch.from_numpy(yy)).float()
                optimizer.zero_grad()
                outputs, penalty = model.forward(input)
                loss = self._loss_function(labels, outputs)
                loss = loss + lamb*penalty
                loss.backward()
                optimizer.step()
                training_accu = np.corrcoef(outputs.detach().numpy().flatten(),
                                            yy.flatten())[0, 1]
                results.iter_accu_training.append(training_accu)
                results.loss.append(loss.item())
            if _ % logging_freq == 0:
                predict, penalty = model.forward(valid_x, False)
                accu = self._accuracy(predict)
                lg.info('Iteration %s: Accuracy: %s Loss: %s',
                        _, accu, loss.item())
                if len(results.iter_accu_valid) > 0:
                    if np.allclose(results.iter_accu_valid[-1], accu, 1e-4):
                        break
                results.add_valid(accu)
                results.add_parameters(model.named_parameters())
        del xx, yy
        predict, penalty = model.forward(valid_x, False)
        accu = self._accuracy(predict)
        coef = {}
        for name, param in model.named_parameters():
            new_name = name.replace('_origin.', '')
            coef[new_name] = param.data.numpy()
        prediction = predict.data.numpy().flatten()
        results.final_results(accu, prediction, coef)
        return results
