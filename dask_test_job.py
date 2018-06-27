import os
import numpy as np
import sklearn.linear_model as lm
import dask.delayed as delayed
from dask.distributed import Client, LocalCluster
from dask_jobqueue import PBSCluster
import scipy.sparse as sparse
from glob import glob


def reader_binary(file):
    mat = sparse.load_npz(file)
    return mat.toarray()


def sum_all(x):
    return np.sum(x, axis=1)


def compute_lasso(X, y, index_valid, alphas):
    X_valid = X[index_valid,:]
    X = np.delete(X, index_valid, axis=0)
    y = np.delete(y, index_valid, axis=0)
    outcomes = []
    for alpha in alphas:
        model = lm.Lasso(alpha=alpha)
        model.fit(X, y)
        prediction = model.predict(X_valid)
        outcomes.append(prediction)
    return np.stack(outcomes).T


if __name__ == '__main__':
    folder = '../data/1kg_LD_blocks/'
    files = glob(folder+'/*')
    index_valid = np.random.randint(0, 1092, 100)
    alphas = np.arange(0.2, 2, 0.2)
    phenoype = np.random.random(1092)

    cluster = PBSCluster(processes=5,
                         threads=4, memory="5GB", project='P48500028',
                         queue='medium',
                         resource_spec='nodes=1:ppa=2:mem=20gb',
                         walltime='02:00:00', interface='ib0')

    client = Client(cluster)

    outcome = []
    for ff in files:
        mat = delayed(reader_binary)(ff)
        pred = delayed(compute_lasso)(mat, phenoype, index_valid, alphas)
        outcome.append(pred)
    combiend_mat = delayed(sum)(outcome)

    newmat = combiend_mat.compute()
