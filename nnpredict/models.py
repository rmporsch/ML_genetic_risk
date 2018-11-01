"""Tensorflow models."""
from nnpredict.decorators import *
import numpy as np
import tensorflow as tf
import logging
import pickle
import os
import re

lg = logging.getLogger(__name__)


class LinearModel(object):

    def __init__(self, X, y,  keep_prob: float,
                 learning_rate: float, penal: float, ld_blocks: str,
                 weights = None):
        self.batch_size, self.p = X.get_shape()
        self.ld_blocks = pickle.load(open(ld_blocks, 'rb'))[10]
        self.bool_blocks = self._make_block_id()
        self.X = X
        self.y = y
        lg.debug('Shape of X: %s \n Shape of y: %s',
                 X.get_shape(),
                 y.get_shape())
        lg.debug('Type of X: %s \n Type of y: %s',
                 X.dtype,
                 y.dtype)
        self.penal = penal
        self.learning_rate = learning_rate
        self.num_blocks = len(self.bool_blocks)
        self.keep_prob = keep_prob
        self.training = tf.cond(keep_prob < 1.0, lambda: tf.constant(True),
                                lambda:tf.constant(False))
        self.streaming_mean_corr = None
        self.weights = weights
        self.dev_status = tf.constant(1.0, name='bool_dev')
        if self.weights is not None:
            self.weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        lg.debug('Batch size is: %s', self.batch_size)
        lg.debug('Learning rate is set to %s', self.learning_rate)
        lg.debug('Number of blocks is set to %s', self.num_blocks)

        self.prediction
        self.cost
        self.optimize
        self.error
        self.error_dev

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

    @define_scope(scope='cost')
    def cost(self):
        mse = tf.losses.mean_squared_error(self.y, self.prediction)
        tf.summary.scalar('MSE', mse)
        penalty = tf.losses.get_regularization_loss()
        tf.summary.scalar('Penalty', penalty)
        cost = tf.add(mse, penalty)
        tf.summary.scalar('Cost', cost)
        return cost

    @define_scope(scope='optimization')
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(self.cost)

    @define_scope(scope='error')
    def error(self):
        x = tf.concat([self.y, self.prediction], axis=1)
        batch_size = tf.cast(tf.shape(x)[0], tf.float32)
        means = tf.reduce_mean(x, axis=0)
        d = tf.pow(tf.subtract(x, means), 2)
        df = tf.reduce_sum(d, axis=0)
        dd = tf.div(df, batch_size)
        sds = tf.sqrt(dd)
        cov = tf.reduce_mean(tf.reduce_prod(tf.subtract(x, means), axis=1))
        corr = tf.div(cov, tf.reduce_prod(sds))
        m, update_m = tf.metrics.mean(corr, name='mean_corr')
        tf.summary.scalar('Correlation', update_m)
        return update_m

    @define_scope(scope='error_dev')
    def error_dev(self):
        x = tf.concat([self.y, self.prediction], axis=1)
        batch_size = tf.cast(tf.shape(x)[0], tf.float32)
        means = tf.reduce_mean(x, axis=0)
        d = tf.pow(tf.subtract(x, means), 2)
        df = tf.reduce_sum(d, axis=0)
        dd = tf.div(df, batch_size)
        sds = tf.sqrt(dd)
        cov = tf.reduce_mean(tf.reduce_prod(tf.subtract(x, means), axis=1))
        corr = tf.div(cov, tf.reduce_prod(sds))
        m, update_m = tf.metrics.mean(corr, name='mean_corr')
        tf.summary.scalar('Correlation', update_m)
        return update_m

    @staticmethod
    def noise(input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer),
                                 stddev=std)
        return input_layer + noise

    @staticmethod
    def tb_layers(layer):
        n = layer.name
        output_name = re.search('LD_block[0-9]*', n).group(0)
        weights = tf.get_default_graph().get_tensor_by_name(
            os.path.split(n)[0]+'/kernel:0')
        mean, var = tf.nn.moments(weights, axes=[0, 1])
        tf.summary.scalar('Mean_'+output_name, mean)
        tf.summary.scalar('Var_'+output_name, var)

    @define_scope(scope='prediction')
    def prediction(self):
        l1 = tf.contrib.layers.l1_regularizer(self.penal)
        initial_values = tf.initializers.random_normal(0.0, 0.0001)
        with tf.variable_scope('LD_blocks'):
            collector = list()
            for i, b in enumerate(self.bool_blocks):
                with tf.variable_scope('LD_block' + str(i)):
                    small_block = tf.boolean_mask(self.X, b, axis=1)
                    small_block.set_shape((self.batch_size, np.sum(b)))
                    lg.debug('Type of small_block: %s', small_block.dtype)
                    y_ = tf.layers.dense(small_block, 1,
                                         kernel_regularizer=l1,
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
                    collector.append(y_)
                    if i % 10 == 0:
                        lg.debug('Made tensors for LD block %s', i)
        collection = tf.concat(collector, name='prediction_matrix', axis=1)
        layer1 = tf.layers.dense(collection, 85, name='layer1',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.relu,
                                 kernel_regularizer=l1)
        drop_out = tf.nn.dropout(layer1, self.keep_prob)
        y_hat = tf.layers.dense(drop_out, 1, name='combinging_linear',
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=l1)
        return y_hat


class NNModel(object):

    def __init__(self, X, y, keep_prob: float, learning_rate: float,
                 penal: float, layers: list):
        self.X = X
        self.y = y
        lg.debug('Shape of X: %s \n Shape of y: %s',
                 X.get_shape(),
                 y.get_shape())
        lg.debug('Type of X: %s \n Type of y: %s',
                 X.dtype,
                 y.dtype)
        self.penal = penal
        self.batch_size, self.dim = X.get_shape()
        layers.insert(0, self.dim)
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.training = tf.cond(keep_prob < 1.0, lambda: tf.constant(True),
                                lambda:tf.constant(False))
        self.layers = layers
        self.streaming_mean_corr = None
        self.dev_status = tf.constant(1.0, name='bool_dev')
        lg.debug('Batch size is: %s', self.batch_size)
        lg.debug('Learning rate is set to %s', self.learning_rate)

        self.prediction
        self.cost
        self.optimize
        self.error

    @define_scope(scope='cost')
    def cost(self):
        mse = tf.losses.mean_squared_error(self.y, self.prediction)
        tf.summary.scalar('MSE', mse)
        penalty = tf.losses.get_regularization_loss()
        tf.summary.scalar('Penalty', penalty)
        cost = tf.add(mse, penalty)
        tf.summary.scalar('Cost', cost)
        return cost

    @define_scope(scope='optimization')
    def optimize(self):
        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        return optimizer.minimize(self.cost)

    @define_scope(scope='error')
    def error(self):
        x = tf.concat([self.y, self.prediction], axis=1)
        batch_size = tf.cast(tf.shape(x)[0], tf.float32)
        means = tf.reduce_mean(x, axis=0)
        d = tf.pow(tf.subtract(x, means), 2)
        df = tf.reduce_sum(d, axis=0)
        dd = tf.div(df, batch_size)
        sds = tf.sqrt(dd)
        cov = tf.reduce_mean(tf.reduce_prod(tf.subtract(x, means), axis=1))
        corr = tf.div(cov, tf.reduce_prod(sds))
        m, update_m = tf.metrics.mean(corr, name='mean_corr')
        tf.summary.scalar('Correlation', update_m)
        return update_m

    @staticmethod
    def noise(input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer),
                                 stddev=std)
        return input_layer + noise

    @staticmethod
    def tb_layers(layer):
        n = layer.name
        output_name = re.search('NN_[0-9]*', n).group(0)
        weights = tf.get_default_graph().get_tensor_by_name(
            os.path.split(n)[0]+'/kernel:0')
        mean, var = tf.nn.moments(weights, axes=[0, 1])
        tf.summary.scalar('Mean_'+output_name, mean)
        tf.summary.scalar('Var_'+output_name, var)

    @define_scope(scope='prediction')
    def prediction(self):
        l1 = tf.contrib.layers.l2_regularizer(self.penal)
        layers = list()
        layers.append(self.X)
        lg.debug('Shape of layers: %s', self.layers)
        with tf.variable_scope('NeuralNetwork'):
            for i in range(len(self.layers)-1):
                lg.debug('Making layer %s', i)
                lg.debug('Shape of input: %s, shape of output %s',
                         layers[i].get_shape(),
                         self.layers[i+1])
                lay = tf.layers.dense(layers[i], self.layers[i+1],
                                      kernel_regularizer=l1,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name='NN_'+str(i))
                self.tb_layers(lay)
                lay_drop = tf.layers.dropout(lay, self.keep_prob)
                layers.append(lay_drop)

            y_hat = tf.layers.dense(layers[-1], 1, name='NN_00_last',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    kernel_regularizer=l1)
            self.tb_layers(y_hat)
        return y_hat
