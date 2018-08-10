"""Tensorflow models."""
from nnpredict.decorators import *
import numpy as np
import tensorflow as tf
import logging

lg = logging.getLogger(__name__)

class LinearModel:

    def __init__(self, X, y, bool_blocks: list, learning_rate: float, penal: float):
        self.X = X
        self.y = y
        self.penal = penal
        self.batch_size = X.get_shape()[0]
        self.learning_rate = learning_rate
        self.bool_blocks = bool_blocks
        self.num_blocks = len(bool_blocks)
        lg.debug('Batch size is: %s', self.batch_size)
        lg.debug('Learning rate is set to %s', self.learning_rate)
        lg.debug('Number of blocks is set to %s', self.num_blocks)

        self.prediction
        self.loss
        self.optimize
        self.error
        self.corr

    @define_scope(scope='loss')
    def loss(self):
        mse = tf.losses.mean_squared_error(self.y, self.prediction)
        tf.summary.scalar('MSE', mse)
        penalty = tf.losses.get_regularization_loss()
        tf.summary.scalar('Penalty', penalty)
        loss = mse + penalty
        tf.summary.scalar('Loss', loss)
        return loss

    @define_scope(scope='optimization')
    def optimize(self):
        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss)

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
        corr = cov / tf.reduce_prod(sds)
        tf.summary.scalar('Error', corr)
        return corr

    # the / makes it re-enter the scope
    @define_scope(scope='error/')
    def corr(self):
        accuracy = tf.contrib.metrics.streaming_pearson_correlation(self.prediction,
                                                                    self.y,
                                                                    name='correlation')
        tf.summary.scalar('Correlation', accuracy[1])
        tf.summary.scalar('Other_Corr', accuracy[0])
        return accuracy

    @define_scope(scope='prediction')
    def prediction(self):
        linear_combiner = tf.constant(1.0, shape=[self.num_blocks, 1])
        rand_norm_init = tf.initializers.random_normal(0, 0.0001)
        collector = list()
        for i, b in enumerate(self.bool_blocks):
            l1 = tf.contrib.layers.l1_regularizer(scale=self.penal, scope=None)
            with tf.variable_scope('LD_block' + str(i)):
                small_block = tf.boolean_mask(self.X, b, axis=1)
                small_block.set_shape((self.batch_size, np.sum(b)))
                y_ = tf.layers.dense(small_block, 1, kernel_regularizer=l1,
                                     kernel_initializer=rand_norm_init)
                collector.append(y_)
                if i % 10 == 0:
                    lg.debug('Made tensors for LD block %s', i)
        collection = tf.concat(collector, name='prediction_matrix', axis=1)
        y_hat = tf.matmul(collection, linear_combiner, name='combinging_linear')
        return y_hat
