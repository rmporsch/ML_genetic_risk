import numpy as np
from sklearn.model_selection import train_test_split
import logging as lg


def get_samples(X, y, index_train, index_valid):
    """Get training and valid samples."""
    assert sum([k in index_train for k in index_valid]) == 0
    n, p = X.shape
    assumed_n = (len(index_valid) + len(index_train))
    if assumed_n == n:
        lg.warning('Seems you did not allocate any testing data')
    elif assumed_n > n:
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


def generate_valid_test_data(n: int, n_valid: float, n_test: float):
    """Give traing, valid and testing index."""
    n_index = np.arange(0, n)
    x_train, x_test = train_test_split(n_index, test_size=n_test)
    x_train, x_valid = train_test_split(x_train, test_size=n_valid)
    return (x_train, x_valid, x_test)
