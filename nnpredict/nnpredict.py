import os
import pandas as pd
import numpy as np
from typing import Any
import logging
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from nnpredict.models import LinearModel
from datetime import datetime

lg = logging.getLogger(__name__)

class NNpredict(object):
    """Neural Network prediction."""

    def __init__(self, ld_blocks: str, pheno_file: str,
                 plink_file: str, batch_files: str):
        super(NNpredict, self).__init__()

        self.batch_files = os.listdir(batch_files)
        self.batch_files = [os.path.join(batch_files, k) for k in self.batch_files]
        self.ld_blocks = pickle.load(open(ld_blocks, 'rb'))[10]
        self.fam = pd.read_table(plink_file+'.fam', sep=' ', header=None)
        self.bim  = pd.read_table(plink_file+'.bim', header=None)
        self.pheno = pd.read_table(pheno_file, sep='\t')
        lg.debug('%s', self.pheno.head())
        self.pheno = self.pheno['V1'].values
        self.p = self.bim.shape[0]
        self.n = self.fam.shape[0]
        self.bool_blocks = self._make_block_id()
        lg.debug('Available subjects in fam file are: %s and we have %s phenotypes',
                 self.n, len(self.pheno))
        lg.debug('Available batch files are %s', len(self.batch_files))
        lg.debug('Available LD blocks are %s', len(self.ld_blocks))
        lg.debug('Phenotype looks like: %s', self.pheno[0:5])


    def _make_block_id(self) -> list:
        output = list()
        u = 0
        for i, b in enumerate(self.ld_blocks):
            nn = len(b)
            mask = np.zeros(self.p, dtype=bool)
            mask[u:(u + nn)] = True
            u += nn
            output.append(mask)
            if i % 10 == 0:
                lg.debug('Processing LD block %s', i)
        return output

    def _geno_iterator(self, paths: list, y: np.ndarray) -> (np.ndarray, np.ndarray):
        np.random.shuffle(paths)
        for p in paths:
            lg.debug('%s', p)
            data, index_vec = np.load(p)
            y_ = y[index_vec]
            n, p = data.shape
            index_shuffle = np.arange(0, n)
            np.random.shuffle(index_shuffle)
            data = data[index_shuffle, :]
            y_ = y_[index_shuffle]
            yield data, y_.reshape(n, 1)

    def set_training_testing(self, test_size: float = None, validation_size: float = None,
                             save_path: str = None, load_path: str = None) -> None:
        """
        Generates separate samples for training, testing and validation.

        Currently it only uses the batch files and does not store the sample ids.

        :param test_size: Proportion of data to use for testing
        :param validation_size: Proportion of the training data to use for validation
        :param save_path: Store testing, training, validation sample to file
        :param load_path: Load previous stored training, validation and testing data
        :return: None
        """

        if load_path is not None:
            with open(load_path, 'rb') as f:
                self.training, self.validation, self.testing = pickle.load(f)
        else:
            assert test_size is not None
            assert validation_size is not None
            self.training, self.testing = train_test_split(self.batch_files,
                                                           test_size=test_size)
            self.training, self.validation = train_test_split(self.training,
                                                              test_size=validation_size)
            if save_path is not None:
                with open(save_path, 'wb') as f:
                    pickle.dump((self.training, self.validation, self.testing), f)
        lg.info('\nTraining: %s \nValidation: %s \nTesting: %s',
                len(self.training), len(self.validation), len(self.testing))

    def _combine_files(self, files):
        geno_iter = self._geno_iterator(files, self.pheno)
        xl = list()
        yl = list()
        for x, y in geno_iter:
            xl.append(x)
            yl.append(y)
        xl = np.concatenate(xl, axis=0)
        yl = np.concatenate(yl, axis=0)
        return xl, yl

    def combine_testing_validation(self, load_path: str = None, save_path: str = None):
        """
        Combines testing and validation data into one matrix and vecotr

        :param load_path: optional path to load the data
        :param save_path: optional path to save the data
        :return:
        """
        if load_path is not None:
            with open(load_path, 'rb') as f:
                datasets = pickle.load(f)
            lg.debug('Loaded files from %s', load_path)
        else:
            datasets = list()
            name = ['validation', 'testing']
            for na, l in zip(name, [self.validation, self.testing]):
                if len(l)==0:
                    datasets.append(None)
                    datasets.append(None)
                    lg.debug('Added 0 to %s', na)
                    continue
                x, y = self._combine_files(l)
                datasets.append(x)
                datasets.append(y)
                lg.debug('Added x: %s and y: %s to %s', x.shape, y.shape, na)
            if save_path is not None:
                with open(save_path, 'wb') as f:
                    pickle.dump(datasets, f)
                lg.debug('Saved file to %s', save_path)
        dat_names = ['x_valid', 'y_valid', 'x_test', 'y_test']
        for na, dat in zip(dat_names, datasets):
            if dat is None:
                lg.debug('Dataset %s has no data', na)
            else:
                lg.debug('Dataset %s has the shape %s', na, dat.shape)
        self.x_valid, self.y_valid, self.x_test, self.y_test = datasets

    def linear_model(self, epochs: int = 400, l_rate: float = 0.001,
                     penal: float = 0.005, name: str = ''):
        """
        Running a linear model.

        :return: None
        """
        assert self.x_valid is not None
        assert self.y_valid is not None
        assert len(self.training) > 1
        Xp = tf.placeholder(tf.float32, [None, self.p], name='genotypes')
        yp = tf.placeholder(tf.float32, [None, 1], name='phenotype')
        keep_prob = tf.placeholder(tf.float32, None, name='dropout_prob')
        now =  datetime.now()
        now = now.strftime('%d/%m/%Y-%H:%M:%S')
        lg.debug('Current time: %s', now)
        model = LinearModel(Xp, yp, self.bool_blocks, keep_prob, l_rate, penal=penal)
        train_writer = tf.summary.FileWriter('tensorboard/neural_network/train-'+name+now,
                                             tf.get_default_graph())
        test_writer = tf.summary.FileWriter('tensorboard/neural_network/test-'+name+now)
        merged_summary = tf.summary.merge_all()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init)
            for i in range(epochs):
                data_iter = self._geno_iterator(self.training, self.pheno)
                for x, y in data_iter:
                    _, loss, summary = sess.run([model.optimize, model.loss, merged_summary],
                                       feed_dict={Xp: x, yp: y, keep_prob: 0.1})
                train_writer.add_summary(summary, i)

                if i % 10 == 0:
                    summary, pred = sess.run([merged_summary, model.prediction],
                                     feed_dict={Xp: self.x_valid, yp: self.y_valid,
                                                keep_prob: 1.0})
                    print(i, np.corrcoef(self.y_valid.flatten(), pred.flatten())[0, 1])
                    test_writer.add_summary(summary, i)
