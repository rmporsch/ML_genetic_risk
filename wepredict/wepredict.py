"""Class wepredict."""
import numpy as np
import sklearn.linear_model as lm
import dask.delayed as delayed
import wepredict.pytorch_regression as pyreg
import dask
import scipy.sparse as sparse
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split

class wepredict(object):
    """Allow predction via LD blocks."""

    def __init__(self, file_path: str, testing: bool):
        """Allow predction via LD blocks."""
        super(wepredict, self).__init__()
        self.files = glob(file_path)
        assert len(self.files) > 0
        self.outcome = None
        if testing:
            self.files = np.random.choice(self.files, 3)

    def reader_binary(self, file):
        mat = sparse.load_npz(file)
        return mat.toarray()

    def sim(self, X):
        effect = np.random.normal(size=X.shape[1])
        index_causal = np.random.randint(0, len(effect),
                                         int(np.floor(len(effect)*0.8)))
        effect[index_causal] = 0
        y = X.dot(effect)
        return y

    def simulate(self):
        """Generate simulated phenotype."""
        simu = []
        for ff in self.files:
            mat = delayed(self._reader_binary)(ff)
            y = delayed(self._sim)(mat)
            simu.append(y)
        overview = delayed(pd.DataFrame.from_items)(
            zip([str(i) for i in range(len(simu))], simu))
        return overview.compute()

    def get_training_valid_test_sample(self, X, y, index_train,  index_valid,
                                       index_test):
        """Get training and valid samples."""
        X_training = X[index_train, :]
        X_valid = X[index_valid, :]
        X_test = X[index_test, :]
        y_training = y[index_train]
        y_valid = y[index_valid]
        y_test = y[index_test]
        return {'training_x': X_training,
                'training_y': y_training,
                'valid_x': X_valid,
                'valid_y': y_valid,
                'test_x': X_test,
                'test_y': y_test}

    def generate_valid_test_data(self, n, n_valid, n_test):
        """Give traing, valid and testing index."""
        n_index = np.arange(0, n)
        x_train, x_test = train_test_split(n_index, test_size=n_test)
        x_train, x_valid = train_test_split(x_train, test_size=n_valid)
        return (x_train, x_valid, x_test)

    def compute_enet(self, X, y, X_valid, y_valid, alphas):
        """Compute Elassitc Net."""
        model = lm.enet_path(X, y, alphas=alphas, X_copy=False)
        outcome = model[1].T.dot(X_valid.T).T
        measure = []
        for i in range(len(alphas)):
            measure.append(np.corrcoef(y_valid, outcome[:, i])[0, 1])
        return {'prediction': outcome, 'model': model,
                'accu': measure, 'pheno': y_valid}

    def compute_lasso(self, X, y, X_valid, y_valid, alphas):
        """Compute Elassitc Net."""
        model = lm.lasso_path(X, y, alphas=alphas, X_copy=False)
        outcome = model[1].T.dot(X_valid.T).T
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
        pred_matrix = [k['pred'] for k in models_pytorch]
        pred_matrix = np.stack(pred_matrix, axis=1)
        accu_alphas = [k['accu'] for k in models_pytorch]
        return {'pred': pred_matrix, 'model': models_pytorch,
                'accu': accu_alphas, 'pheno': y_valid}

    def generate_DAG(self, phenotype, index_valid, alphas, norm):
        """Get future objects."""
        outcome = []
        for ff in self.files:
            mat = delayed(self._reader_binary)(ff)
            sample = delayed(self._get_training_valid_sample)(mat, phenotype,
                                                              index_valid)
            pred = delayed(self.compute_pytorch)(sample['training_x'],
                                                 sample['training_y'],
                                                 sample['valid_x'],
                                                 sample['valid_y'],
                                                 alphas, norm)
            outcome.append(pred)
        self.outcome = outcome

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

    def compute(self):
        """Compute DAG."""
        assert self.outcome is not None
        return dask.compute(self.outcome)[0]
