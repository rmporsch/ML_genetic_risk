"""Class wepredict."""
import numpy as np
import sklearn.linear_model as lm
import dask.delayed as delayed
import dask
from wepredict.plink_reader import Genetic_data_read
import wepredict.pytorch_regression as pyreg
import scipy.sparse as sparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

lg = logging.getLogger(__name__)

class wepredict(Genetic_data_read):
    """Allow predction via LD blocks."""

    def __init__(self, plink_file: str, ld_blocks: str,
                 pheno: str = None, testing: bool = False):
        """Allow predction via LD blocks."""
        super().__init__(plink_file, ld_blocks, pheno)
        if testing:
            rand_chr = np.random.choice(list(self.groups.keys()), 1)[0]
            lg.debug('Chromosom randomly selected %s', rand_chr)
            lg.debug('len: %s and type of group %s with %s items',
                     len(self.groups),
                     type(self.groups[rand_chr]), len(self.groups[rand_chr]))
            num_blocks = len(self.groups[rand_chr])
            rand_blocks = np.array(np.random.choice(range(num_blocks), 3,
                                           replace=False))
            lg.debug('type of random block is %s of size %s and looks like %s',
                     type(rand_blocks), len(rand_blocks), rand_blocks)
            rand_groups = np.array(self.groups[rand_chr])[rand_blocks]
            self.blocks = {rand_chr: rand_groups}
        else:
            self.blocks = self.groups
        assert sum([len(k) for k in self.blocks.items()]) > 0

    def reader_binary(self, file):
        """Read a binary sparse matrix file."""
        mat = sparse.load_npz(file)
        return mat.toarray()

    def _sim(self, X):
        """Simulate on a genotype matrix."""
        effect = np.random.normal(size=X.shape[1])
        index_causal = np.random.randint(0, len(effect),
                                         int(np.floor(len(effect)*0.8)))
        effect[index_causal] = 0
        y = X.dot(effect)
        return y

    def simulate(self):
        """Generate simulated phenotype."""
        simu = list()
        chr = self.blocks[22]
        for block in chr:
            print(block)
            mmat = self.block_read(22, block)
            print(mmat.shape)
            mat = delayed(self.block_read)(22, block)
            y = delayed(self._sim)(mat)
            simu.append(y)
        return simu

    def get_samples(self, X, y, index_train, index_valid):
        """Get training and valid samples."""
        assert sum([k in index_train for k in index_valid]) == 0
        assumed_n = (len(index_valid) + len(index_train))
        if assumed_n == self.n:
            lg.warning('Seems you did not allocate any testing data')
        elif assumed_n > self.n:
            raise ValueError('More samples in index than in bed file!')
        else:
            lg.info('Allocated %s for training and %s for validation', len(index_train),
                    len(index_valid))
        X_training = X[index_train, :]
        X_valid = X[index_valid, :]
        y_training = y[index_train]
        y_valid = y[index_valid]
        return {'training_x': X_training,
                'training_y': y_training,
                'valid_x': X_valid,
                'valid_y': y_valid}

    def generate_valid_test_data(self, n_valid, n_test):
        """Give traing, valid and testing index."""
        n_index = np.arange(0, self.n)
        x_train, x_test = train_test_split(n_index, test_size=n_test)
        x_train, x_valid = train_test_split(x_train, test_size=n_valid)
        return (x_train, x_valid, x_test)

    def compute_enet(self, X, y, X_valid, y_valid, alphas):
        """Compute Elassitc Net."""
        model = lm.enet_path(X, y, alphas=alphas, X_copy=False)
        alphas, coefs, dual_gaps = model
        outcome = X_valid.dot(coefs)
        measure = []
        for i in range(len(alphas)):
            measure.append(np.corrcoef(y_valid, outcome[:, i])[0, 1])
        return {'prediction': outcome, 'model': model,
                'accu': measure, 'pheno': y_valid}

    def compute_lasso(self, X, y, X_valid, y_valid, alphas):
        """Compute Elassitc Net."""
        model = lm.lasso_path(X, y, alphas=alphas, X_copy=False)
        alphas, coefs, dual_gaps = model
        outcome = X_valid.dot(coefs)
        measure = []
        for i in range(len(alphas)):
            measure.append(np.corrcoef(y_valid, outcome[:, i])[0, 1])
        return {'pred': outcome, 'model': model,
                'accu': measure, 'pheno': y_valid}

    def compute_pytorch(self, X, y, X_valid, y_valid, alphas, norm,
                        mini_batch=250, l_rate=0.001, epochs=201):
        """Pytorch Lo."""
        models_pytorch = list()
        model = pyreg.pytorch_linear(X, y, X_valid, y_valid, type='c',
                                     mini_batch_size=mini_batch)
        for a in alphas:
            model_output = model.run(penal=norm, epochs=epochs, l_rate=l_rate,
                                     lamb=float(a))
            models_pytorch.append(model_output)
        return models_pytorch

    def generate_DAG(self, phenotype, train_index, valid_index,
                     alphas, norm, **kwargs):
        """Get future objects."""
        outcome = {}
        for chr in self.blocks:
            block_outcome = []
            for block in chr:
                mat = delayed(self.block_read)(int(chr), block)
                sample = delayed(self._get_sample)(mat, phenotype, train_index,
                                                   valid_index)
                pred = delayed(self.compute_pytorch)(sample['training_x'],
                                                     sample['training_y'],
                                                     sample['valid_x'],
                                                     sample['valid_y'],
                                                     alphas, norm, **kwargs)
                block_outcome.append(pred)
            outcome[chr] = block_outcome
        return outcome.compute()

    def generate_DAG_testing(self, test_pheno, test_index, best_model):
        """Write DAG for evaluating coef on the test data."""
        outcome = {}
        assert isinstance(best_model, dict)
        for chr in self.blocks:
            block_outcome = []
            for i, block in enumerate(chr):
                coef = best_model[chr][i]
                mat = delayed(self.p_reader.block_iter)(int(chr), block)
                test_mat = mat[test_index, :]
                block_performance = self.evaluate_test(test_mat,
                                                       test_pheno, coef)
                block_outcome.append(block_performance)
            outcome[chr] = block_outcome
        return outcome.compute()

    def evaluate_test(self, X_test, y_test, coef):
        """Evaluate performance on test data."""
        y_pred = X_test.dot(coef.T).flatten()
        accu = np.corrcoef(y_pred, y_test)[0, 1]
        return {'accu': accu, 'pred': y_pred}

    def evaluat_blocks_valid(self, outcome, valid_pheno):
        """Evaluate performance of the combined LD blocks."""
        # assert isinstance(outcome, list)
        combined_prediction = sum([k['pred'] for k in outcome])
        n_blocks = len(outcome)
        n_valid_pheno = len(valid_pheno)
        n_valid_x, n_alphas = combined_prediction.shape
        assert n_valid_pheno == n_valid_x
        # evaluate overall performance
        alpha_eval = []
        for i in range(n_alphas):
            alpha_eval.append(np.corrcoef(combined_prediction[:, i],
                                          valid_pheno)[0, 1])
        block_performance = zip([str(i) for i in range(n_blocks)],
                                np.array([k['accu'] for k in outcome]))
        block_performance = pd.DataFrame.from_items(block_performance)
        return {'Overall': alpha_eval, 'Block': block_performance}
