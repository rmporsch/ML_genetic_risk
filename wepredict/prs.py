"""Compute polygenic risk scores."""
import numpy as np
import sklearn.linear_model as linear
import os
import pickle
import scipy.stats as stats
import pandas as pd
import datetime
from pprint import pprint
from typing import Any
from distutils.spawn import find_executable
import glob
from subprocess import Popen


class Plink(object):
    """docstring for Plink."""

    def __init__(self, plinkfile, plink_executable=''):
        """Plink."""
        super(Plink, self).__init__()
        self.plinkfile = plinkfile
        if len(plink_executable) > 0:
            self.plink_executable = plink_executable
            assert os.path.isfile(self.plink_executable)
        else:
            self.plink_executable = find_executable('plink')
            assert os.path.isfile(self.plink_executable)

    def clumping(self, summary_stats, p1=0.0001,
                 p2=0.01, r2=0.5, kb=250,
                 snp_field='SNP', p_field='P'):
        """
        Perform clumping with given summary stats.

        :summary_stats: summary statistics (SNP and P value column required)
        :p1: index variant p-value threshold
        :p2: clumped variant p-value threshold
        :r2: r^2 threshold
        :kp: clump kb radius
        :returns: dataframe with clumped SNPs
        """
        output_location = '/tmp/plink_clump'

        if os.path.isfile('/tmp/plink_clump.log'):
            for filename in glob.glob('/tmp/plink_clump*'):
                os.remove(filename)
        summary_stats.to_csv(output_location+'.report', index=False, sep=' ')
        assert len(summary_stats.columns.values) > 2
        if snp_field not in summary_stats.columns.values:
            raise ValueError('SNP filed not in summary stats object')
        if p_field not in summary_stats.columns.values:
            raise ValueError('Pvalue filed not in summary stats object')

        with open(os.devnull, 'w') as fp:
            plink_run = Popen(
                [self.plink_executable,
                 '--bfile', self.plinkfile,
                 '--allow-no-sex',
                 '--clump', output_location+'.report',
                 '--clump-p1', str(p1),
                 '--clump-p2', str(p2),
                 '--clump-r2', str(r2),
                 '--clump-kb', str(kb),
                 '--clump-snp-filed', snp_field,
                 '--clump-filed', p_field,
                 '--out', output_location], stdout=fp)
            plink_run.wait()

        results = pd.read_table(output_location+'.clumped',
                                delim_whitespace=True)
        return results


class polygenic_model(object):
    """Simple polygenic model with p-value threshold."""

    def __init__(self, X, y, model_log, plink_file, overwrite=False, type='b'):
        """Polygenic model with p-value threshold."""
        super(polygenic_model, self).__init__()
        self.model_log = model_log
        self.n, self.input_dim = X.shape
        self.overwrite = overwrite
        self._shuffle_ids = np.arange(self.n)
        np.random.shuffle(self._shuffle_ids)
        self.output_dim = 1
        self.type = type
        self.X = X[self._shuffle_ids, :]
        self.y = y[self._shuffle_ids].reshape(self.n, 1)
        self.gwas = None
        self.plink = Plink(plink_file)
        print("init X shape", self.X.shape)
        if not os.path.isfile(model_log):
            with open(model_log, 'w', encoding='utf-8') as f:
                pickle.dump([], f)
        elif overwrite:
            with open(model_log, 'wb') as f:
                pickle.dump([], f)
        assert os.path.isfile(self.model_log)

    def _compute_varb(self, y_, y, x):
        sse = np.sum((y_ - y)**2, axis=0) / float(self.n - 2)
        varb = sse / np.sum(x - np.mean(x))
        return varb

    def _gwas(self):
        if self.type == 'c':
            model = linear.LinearRegression(fit_intercept=False)
        elif self.type == 'b':
            model = linear.LogisticRegression(C=1e99, fit_intercept=False)
        else:
            raise ValueError('type has to be either c or b')

        out = []
        for i in range(self.input_dim):
            model.fit(self.X[: i], self.y)
            coef = model.coef_
            assert len(coef) == 1
            y_ = model.predict(self.X[:, i])
            tvalue = coef / self._compute_varb(y_, self.y, self.X[:, i])
            pvalue = 2 * (1 - stats.t.cdf(np.abs(tvalue), self.n - 1))
            out.append({'id': i, 'coef': coef[0],
                        'tvalue': tvalue, 'pvalue': pvalue})
        self.gwas_results = pd.from_dict(out)

    def _write_model(self, param: dict, coef: Any,
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

    def _polygenic_score(self, index, coef):
        y_ = np.dot(self.X[: index], coef)
        return np.corr(y_, self.y)

    def run(self, lamb):
        """Run Pvalue threshold based polygenic model."""
        if self.gwas_results is None:
            self._gwas()
            self.gwas_results = self.plink.clumping(self.gwas_results)
        filtered_results = self.gwas_results[self.gwas_results.P <= lamb]

        score = self._polygenic_score(filtered_results.index.values,
                                      filtered_results.coef.values)
        show_output = {
            'Model': 'PRS GWAS linear',
            'time': str(datetime.datetime.now()),
            'score': score,
            'type': self.type,
            'coef': filtered_results.coef.values
        }
        pprint(show_output)
        param = {'threshold': lamb}
        self._write_model(param, filtered_results.coef.values,
                          score, 'pvalue-threshold')
