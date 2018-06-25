"""L1 and L2 norm as implemented in sklearn."""
import sklearn.linear_model as lm
import numpy as np
import os
import pprint
import datetime
import pickle


class sklearn_models(object):
    """Sklearn models."""

    def __init__(self, X: np.array, y: np.array,
                 model_log: str, overwrite: bool = False,
                 type: str = 'b', mini_batch_size=100):
        """Sklearn models."""
        super(sklearn_models, self).__init__()
        self.model_log = model_log
        self.overwrite = overwrite
        self.X = X
        self.y = y
        self.type = type
        self.mini_batch_size = mini_batch_size
        if not os.path.isfile(model_log):
            with open(model_log, 'w', encoding='utf-8') as f:
                pickle.dump([], f)
        elif overwrite:
            with open(model_log, 'wb') as f:
                pickle.dump([], f)
        assert os.path.isfile(self.model_log)

    def _write_model(self, param: dict, coef: np.array,
                     score: float, model_name: str):
        output = {}
        output['param'] = param
        output['coef'] = coef
        output['score'] = score
        output['time'] = str(datetime.datetime.now())
        output['name'] = model_name
        feed = pickle.load(open(self.model_log, 'rb'))
        with open(self.model_log, 'wb') as f:
            feed.append(output)
            pickle.dump(feed, f)

    def _SKLinearRegression(self, penalty: str, lamb: float):
        if penalty == 'l1':
            return lm.Lasso(alpha=lamb)
        elif penalty == 'l2':
            return lm.Ridge(alpha=lamb)
        else:
            raise ValueError('needs to be l1 or l2')

    def run(self, penal: str = 'l1', lamb: float = 0.01):
        """Run sklearn regression."""
        if self.type == 'b':
            model = lm.LogisticRegression(penalty=penal, C=lamb)
        elif self.type == 'c':
            model = self._SKLinearRegression(penalty=penal, lamb=lamb)
        else:
            raise ValueError('type has to be either b or c')

        model.fit(self.X, self.y)
        param = model.get_params()
        coef = model.coef_
        score = model.score(self.X, self.y)
        show_output = {
            'Model': 'L1 Norm sklearn',
            'time': str(datetime.datetime.now()),
            'score': score,
            'Num. of non-zero coef': np.sum(coef != 0),
            'type': self.type}
        pprint.pprint(show_output)
        self._write_model(param, coef, score, penal+' Norm sklearn')
