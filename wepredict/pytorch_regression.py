"""Class and function for penalized regressions with tensorflow."""
import numpy as np
import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import scale
import logging
from wepredict.l0_norm import L0Linear
from sklearn.utils import shuffle

lg = logging.getLogger(__name__)

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

    def __init__(self, train_iter, dev_iter, input_dim,
                 n_samples, dev_num_iter, batch_size, type='b'):
        """
        Linear regression with L0 or L1.

        :param train_iter: iterator for the training data
        :param dev_iter:  iterator for the dev dta
        :param input_dim: input dimensions
        :param n_samples: number of samples
        :param dev_num_iter: number of tieration for dev data
        :param type: binary ('b') or continious ('c') trait
        """
        super(pytorch_linear, self).__init__()
        self.input_dim = input_dim
        self.n = n_samples
        self.dev_num_iter = dev_num_iter
        self.output_dim = 1
        self.type = type
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.batch_size = batch_size

    def _model_builder(self, penal, **kwargs):
        if penal == 'l1':
            model = RegL1(self.input_dim, self.output_dim)
        elif penal == 'l0':
            model = RegL0(self.input_dim, self.output_dim)
        elif penal == 'l02':
            model = RegL0(self.input_dim, self.output_dim, l02=True)
        else:
            raise ValueError('incorrect norm name specified')
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

    def _accu_dev(self, model):
        corrs = []
        for _ in range(self.dev_num_iter):
            xx, yy = next(self.dev_iter)
            valid_x = Variable(torch.from_numpy(xx)).float()
            predict, penalty = model.forward(valid_x, False)
            training_accu = np.corrcoef(predict.data.numpy().flatten(),
                                        yy.flatten())[0, 1]
            corrs.append(training_accu)
        return np.mean(corrs)


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

        for _ in range(epochs):
            one_iter = False; u = 0
            while one_iter is False:
                xx, yy = next(self.train_iter)
                input = Variable(torch.from_numpy(xx)).float()
                labels = Variable(torch.from_numpy(yy)).float()
                optimizer.zero_grad()
                outputs, penalty = model.forward(input)
                loss = self._loss_function(labels, outputs)
                loss = loss + lamb*penalty
                loss.backward()
                optimizer.step()
                training_accu = np.corrcoef(outputs.data.numpy().flatten(),
                                            yy.flatten())[0, 1]
                results.iter_accu_training.append(training_accu)
                lg.debug('Current training accu at %s: %s', u,
                         training_accu)
                results.loss.append(loss.item())
                if u > self.n:
                    one_iter = True
                u += self.batch_size
            if _ % logging_freq == 0:
                accu = self._accu_dev(model)
                lg.info('Iteration %s: Accuracy: %s', _, accu)
                results.add_valid(accu)
                results.add_parameters(model.named_parameters())
        accu = self._accu_dev(model)
        lg.debug('Last dev accu is %s', accu)
        coef = {}
        for name, param in model.named_parameters():
            new_name = name.replace('_origin.', '')
            coef[new_name] = param.data.numpy()

        results.final_results(accu, [], coef)
        return results
