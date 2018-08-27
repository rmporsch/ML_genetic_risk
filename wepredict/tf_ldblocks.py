import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import tensorboard as tb
import logging
import os
from pyplink import PyPlink
from sklearn.preprocessing import scale

lg = logging.getLogger(__name__)

class NNmodel(object):

    def __init__(self):
        super(NNmodel, self).__init__()

    def get_block_matrix(self, data, block):
        num_rows, num_cols = data.shape
        num_features = np.sum(block)
        new_block = tf.boolean_mask(data, block, axis=1)
        new_block = tf.reshape(new_block, shape=(num_rows, num_features))
        lg.debug('size of new_block is %s with type %s', new_block.shape, new_block.dtype)
        return new_block

    def linear(self, in_var, num_var):
        lg.debug('inpyt type is: %s', in_var.dtype)
        weights = tf.get_variable('weights', [num_var, 1],
                                  initializer=tf.random_normal_initializer(0, 0.0001))
        bias = tf.get_variable('bias', [1, 1],
                               initializer=tf.constant_initializer(0.0))
        comb = tf.matmul(in_var, weights)
        return tf.nn.relu(comb + bias)

    def linear_block_wise(self, data, blocks):
        linear_block_variables = list()
        for i, b in enumerate(blocks):
            p = sum(b)
            lg.debug('processing LD block %s wiht %s SNPs', i, p)
            with tf.variable_scope('linear_block_' + str(i)):
                block_matrix = self.get_block_matrix(data, b)
                linear_block_variables.append(self.linear(block_matrix, p))
        return linear_block_variables

    def sum_linear(self, data, blocks):
        linear_block_variables = list()
        for i, b in enumerate(blocks):
            p = sum(b)
            lg.debug('processsing LD block %s with %s SNPs', i, p)
            with tf.variable_scope('linear_block_'+str(i)):
                block_matrix = self.get_block_matrix(data, b)
                weights = tf.get_variable('weights', [p, 1],
                                          initializer=tf.random_normal_initializer(0, 0.0001))
                bias = tf.get_variable('bias', [1, 1],
                                   initializer=tf.constant_initializer(0.0))
                y_hat = tf.matmul(block_matrix, weights) + bias
                linear_block_variables.append(y_hat)

        all_y_hat = tf.add_n(linear_block_variables)
        return all_y_hat

    def add_layers_on_blocks(self, ld_blocks_weights, num_layers):
        n_blocks = len(ld_blocks_weights)
        stacked_blocks = tf.concat(ld_blocks_weights, 1)
        lg.debug('stacked blocks are %s', stacked_blocks.shape)
        with tf.variable_scope('layer_over_blocks'):
            weights = tf.get_variable('weights',
                                      [n_blocks, num_layers],
                                      initializer=tf.random_normal_initializer(0, 0.0001))
            bias = tf.get_variable('bias', [num_layers],
                                   initializer=tf.constant_initializer(0.0))
            comb = tf.nn.bias_add(tf.matmul(stacked_blocks, weights), bias)
        return tf.nn.relu(comb)

    def add_layers(self, origin, num_layers, scope_name, output=False):
        with tf.variable_scope(scope_name):
            n, p = origin.shape
            weights = tf.get_variable('weights',
                                      [p, num_layers],
                                      initializer=tf.random_normal_initializer())
            bias = tf.get_variable('bias', [num_layers],
                                   initializer=tf.constant_initializer(0.0))
            comb = tf.nn.bias_add(tf.matmul(origin, weights), bias)
        if output:
            return comb
        else:
            return tf.nn.relu(comb)

class NNpredict(NNmodel):

    def __init__(self, num_snps: int, mini_batch_size: int, path: str):
        super(NNpredict, self).__init__()
        self.num_snps = num_snps
        self.mini_batch_size = mini_batch_size
        self.batch_files = os.listdir(path)
        self.batch_files = [os.path.join(path, k) for k in self.batch_files]
        self.input_placeholder = tf.placeholder(tf.float32, [mini_batch_size, num_snps])
        self.y_placeholder = tf.placeholder(tf.float32, [mini_batch_size, 1])
        lg.info('There are %s file present', len(self.batch_files))

    def make_block_id(self, blocks):
        output = list()
        u = 0
        for i, b in enumerate(blocks):
            nn = len(b)
            mask = np.zeros(self.num_snps, dtype=bool)
            mask[u:(u + nn)] = True
            u += nn
            output.append(mask)
            if i % 10 == 0:
                lg.debug('Processing LD block %s', i)
        return output


    def data_iter(self, pheno, shuffle_values=True):
        np.random.shuffle(self.batch_files)
        for p in self.batch_files:
            dat, bool_index = np.load(p)
            lg.debug('current size of block is %s of type %s',
                     np.sum(bool_index), type(bool_index))
            dat = dat.astype(np.float32)
            batch_pheno = pheno[bool_index].astype(float)
            batch_pheno = batch_pheno.reshape(len(batch_pheno), 1)
            if shuffle_values:
                dat, batch_pheno = shuffle(dat, batch_pheno)
            yield scale(dat), batch_pheno, p

    def sum_regression(self, ld_blocks, pheno, epochs=1, save_path=None):
        blocks = self.make_block_id(ld_blocks)
        y_hat = self.sum_linear(self.input_placeholder, blocks)
        mse = tf.reduce_mean(tf.square(self.y_placeholder - y_hat))
        optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(mse)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(100):
                iterator = self.data_iter(pheno)
                u = 0
                for x, y, path in iterator:
                    _, c = sess.run([optimizer, mse],
                                    feed_dict={self.input_placeholder: x,
                                               self.y_placeholder: y})
                    assert np.isfinite(c)
                    if u % 10 == 0:
                        print(c)
#     def build_nn(self, num_layers, layer_shape, blocks):
#         assert len(layer_shape) == num_layers
#         assert layer_shape[-1] == 1
#         tf.reset_default_graph()
#         bool_blocks = self.make_block_id(blocks)
#         input_data = tf.placeholder(tf.float32,
#                                     [self.mini_batch_size, self.num_snps])
#         output_data = tf.placeholder(tf.float32,
#                                      [self.mini_batch_size, 1])
#         block_layer = self.linear_block_wise(input_data, bool_blocks)
#
#         layers = list()
#         for i, u in enumerate(layer_shape):
#             if i == (len(num_layers)-1):
#                 layers.append(self.add_layers(layers[-1], 1, output=True))
#             elif i == 0:
#                 layers.append(self.add_layers(block_layer, u))
#             else:
#                 layers.append(self.add_layers(layers[-1], u))


