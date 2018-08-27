"""Tensorflow models."""
from nnpredict.decorators import *
import numpy as np
import tensorflow as tf
import logging

lg = logging.getLogger(__name__)

class LinearModel:

    def __init__(self, X, y, bool_blocks: list, keep_prob: float,
                 learning_rate: float, penal: float):
        self.X = X
        self.y = y
        self.penal = penal
        self.batch_size = X.get_shape()[0]
        self.learning_rate = learning_rate
        self.bool_blocks = bool_blocks
        self.num_blocks = len(bool_blocks)
        self.keep_prob = keep_prob
        self.training = tf.cond(keep_prob < 1.0, lambda: tf.constant(True),
                                lambda:tf.constant(False))
        self.streaming_mean_corr = None
        lg.debug('Batch size is: %s', self.batch_size)
        lg.debug('Learning rate is set to %s', self.learning_rate)
        lg.debug('Number of blocks is set to %s', self.num_blocks)

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

    @define_scope(scope='prediction')
    def prediction(self):
        rand_norm_init = tf.initializers.random_normal(0.0, 0.0001)
        rand_norm_init_1 = tf.initializers.random_normal(0.0, 0.0001)
        l1 = tf.contrib.layers.l1_l2_regularizer(scale_l1=self.penal,
                                                 scale_l2=self.penal, scope=None)
        with tf.variable_scope('LD_blocks'):
            collector = list()
            for i, b in enumerate(self.bool_blocks):
                with tf.variable_scope('LD_block' + str(i)):
                    small_block = tf.boolean_mask(self.X, b, axis=1)
                    small_block.set_shape((self.batch_size, np.sum(b)))
                    y_ = tf.layers.dense(small_block, 1, kernel_regularizer=l1,
                                         kernel_initializer=rand_norm_init)
                    collector.append(y_)
                    if i % 10 == 0:
                        lg.debug('Made tensors for LD block %s', i)
        collection = tf.concat(collector, name='prediction_matrix', axis=1)
        norm_collection = tf.layers.batch_normalization(collection, axis=1, center=True,
                                                        training=self.training)
        drop_out = tf.nn.dropout(norm_collection, self.keep_prob)
        y_hat = tf.layers.dense(drop_out, 1, name='combinging_linear',
                               kernel_initializer=rand_norm_init_1,
                               kernel_regularizer=l1)
        return y_hat
