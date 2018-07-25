import numpy as np
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt
import logging as lg
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from wepredict.pytorch_regression import pytorch_linear

# lg.basicConfig(level=lg.DEBUG)


def sim(n, p, null_prop):
    lg.info('n: %s, p: %s', n, p)
    pred = list()
    for i in range(p):
        pred.append(np.random.normal(0, 1, n))
    pred = np.column_stack(pred)
    beta = np.random.normal(0, 0.1, p)
    null_index = np.random.choice(range(p),int(p*null_prop),
                     replace=False)
    beta[null_index] = 0
    y = scale(pred.dot(beta.T))
    y = scale(y) + np.random.normal(0, 1, n)
    return y, pred

def sklearn_model(x, y, x_valid, y_valid, alphas):
    model = lasso_path(x, y, alphas=alphas)
    y_hat = x_valid.dot(model[1])
    corr_valid = list()
    for k in range(len(alphas)):
        corr_valid.append(np.corrcoef(y_valid, y_hat[:, k])[0, 1])
    best_lambda = np.argmax(corr_valid)
    lg.debug('prediction value for alphas: %s', corr_valid)
    lg.debug('position of best value: %s', best_lambda)
    return corr_valid[best_lambda]

def torch_model(x_train, y_train, x_valid, y_valid, alphas, regu):
    torch_model = pytorch_linear(x_train, y_train, x_valid, y_valid,
                                 type='c', mini_batch_size=50)
    outcome = list()
    for a in alphas:
        outcome.append(torch_model.run(regu, float(a), epochs=110))

    predictions = list()
    for i in outcome:
        predictions.append(i.accu)
    best_lamb = np.argmax(predictions)
    lg.debug('best lambda: %s', best_lamb)
    return outcome[best_lamb]


if __name__ == '__main__':
    n = 100000
    p = 1000
    null_prop = 0.98
    y, pred = sim(n, p, null_prop)
    # alpha_values = np.arange(0.001, 0.01, 0.001)
    alpha_values = [0.5]
    lg.info('Number of alphas: %s', alpha_values)
    x_train, x_test, y_train, y_test = train_test_split(pred, y,
                                                        test_size=0.1,
                                                        random_state=42)
    del pred
    # skoutput = sklearn_model(x_train, y_train, x_test, y_test, alpha_values)
    # print('sklearn best:', skoutput)

    regus = ['l0', 'l1']
    sample_limit = np.logspace(2, 5, 10, dtype=int)
    out = dict()
    out['l1'] = list()
    out['l0'] = list()
    for s in sample_limit:
        for l in regus:
            alpha_values = [2.0] if l == 'l0' else [0.1]
            torch_out = torch_model(x_train[0:int(s), :], y_train[0:int(s)],
                                    x_test, y_test, alpha_values, regu=l)
            out[l].append(torch_out)
            print('Best Torch model', l, ':', torch_out.accu, 'with', s)

