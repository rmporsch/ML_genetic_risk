"""Testing pytorch."""
from wepredict import pytorch_regression as pyreg
import numpy as np
import scipy.sparse as sparse
from matplotlib import pyplot as plt

def sim(X):
    effect = np.random.normal(size=X.shape[1])
    index_causal = np.random.randint(0, len(effect),
                                     int(np.floor(len(effect)*0.8)))
    effect[index_causal] = 0
    y = X.dot(effect)
    return y

def reader_binary(file):
    mat = sparse.load_npz(file)
    return mat.toarray()

def get_training_valid_sample(X, y, index_valid):
    mask = np.ones(len(y), dtype=bool)
    mask[index_valid] = False
    return (X[mask, :], y[mask], X[~mask, :], y[~mask])

def compute_pytorch_l1(X, y, X_valid, y_valid, alphas, mini_batch_size=250):
    model = pyreg.pytorch_linear(X, y, X_valid, y_valid, True, type='c',
                                 mini_batch_size=mini_batch_size)
    model_output = model.run(penal='l1', epochs=501, l_rate=0.001)
    return model_output

def compute_pytorch_l2(X, y, X_valid, y_valid, alphas, mini_batch_size=250):
    model = pyreg.pytorch_linear(X, y, X_valid, y_valid, True, type='c',
                                 mini_batch_size=mini_batch_size)
    model_output = model.run(penal='l2', epochs=501, l_rate=0.001)
    return model_output

def compute_pytorch_l0(X, y, X_valid, y_valid, alphas, mini_batch_size=250):
    model = pyreg.pytorch_linear(X, y, X_valid, y_valid, True, type='c',
                                 mini_batch_size=mini_batch_size)
    model_output = model.run(penal='l0', epochs=501, l_rate=0.001)
    return model_output

if __name__ == '__main__':
    testfile = 'data/1kg_LD_blocks/22_LD_block_0.npz'
    index_valid_test = np.random.randint(0, 1092, 100)
    alphas = np.arange(0.01, 0.2, 0.02)
    X = reader_binary(testfile)
    y = sim(X)
    sample = get_training_valid_sample(X, y, index_valid_test)
    model_output_l1 = compute_pytorch_l1(sample[0], sample[1], sample[2],
                                      sample[3], alphas, mini_batch_size=400)
    model_output_l2 = compute_pytorch_l2(sample[0], sample[1], sample[2],
                                      sample[3], alphas, mini_batch_size=400)
    model_output_l0 = compute_pytorch_l0(sample[0], sample[1], sample[2],
                                      sample[3], alphas, mini_batch_size=400)
    print('L1', model_output_l1['accu'])
    print('L2', model_output_l2['accu'])
    print('L0', model_output_l0['accu'])
    # plot Sigmoid
    x = [x for x in range(model_output_l1['param']['epoch'])]
    fig, ax = plt.subplots()
    ax.plot(x, model_output_l1['param']['loss'], 'r')
    ax.plot(x, model_output_l2['param']['loss'], 'b')
    ax.plot(x, model_output_l0['param']['loss'], 'y')
    plt.show()
