"""
Use class for predicting phenotype from genotypes
"""
from pyplink_major.plink_reader import Major_reader
from wepredict.pytorch_regression import pytorch_linear
import logging

lg = logging.getLogger(__name__)

class Predict(object):

    def __init__(self, train_path: str, dev_path: str, pheno: str,
                 batch_size: int, pheno_dev: str = None):
        super(Predict, self).__init__()
        self._plink_train_path = train_path
        self._plink_dev_path = dev_path
        self.train = Major_reader(train_path, pheno)
        assert (self.train.n / batch_size).is_integer()
        if pheno_dev is None:
            self.dev = Major_reader(dev_path, pheno)
        else:
            self.dev = Major_reader(dev_path, pheno_dev)
        assert (self.dev.n / batch_size).is_integer()
        self.batch_size = batch_size
        self.results = None
        self.num_dev_iter = int(self.dev.n / batch_size)

        lg.info('Using %s for training and %s for devop. Mini-batch size for both is set to %s',
                self.train.n, self.dev.n, batch_size)

    def fit(self, pheno: str, penal: str, lamb: float,
            l_rate: float, epochs: int = 201,
            logging_freq: int = 100, type: str = 'c'):
        assert pheno in self.train.pheno_names
        assert pheno in self.dev.pheno_names

        train_reader = self.train.read(pheno, self.batch_size)
        dev_reader = self.dev.read(pheno, self.batch_size)
        lg.debug('Finished setting up the iterators')
        model = pytorch_linear(train_reader, dev_reader, self.train.p, self.train.n,
                               self.num_dev_iter, self.batch_size, type)
        lg.debug('Set up linear model')
        self.results = model.run(penal, lamb, epochs, l_rate, logging_freq)
        lg.debug('Model finished')
