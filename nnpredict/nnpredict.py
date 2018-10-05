import os
import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from nnpredict.models import LinearModel
from datetime import datetime
from pyplink_major import plink_reader as pr

lg = logging.getLogger(__name__)

class NNpredict(object):
    """Neural Network prediction."""

    def __init__(self, plink_file_train: str, plink_file_dev: str,
                 pheno_file: str, ld_blocks: str,
                 tb_path: str = 'tensorboard/neural_network/'):
        super(NNpredict, self).__init__()
        self.ld_blocks = pickle.load(open(ld_blocks, 'rb'))[10]
        self.dtrain = pr.Major_reader(plink_file_train, pheno_file, ld_blocks)
        self.ddev = pr.Major_reader(plink_file_dev, pheno_file, ld_blocks)
        assert self.dtrain.p == self.ddev.p
        self.p = self.ddev.p
        self.n = self.dtrain.n
        tb_path = 'tensorboard/neural_network/'
        self.bool_blocks = self._make_block_id()
        self.tb_path = tb_path
        lg.debug('Available LD blocks are %s', len(self.ld_blocks))


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

    def _make_dataset(self, pheno_name: str):
        dd = tf.data.Dataset()
        output_shapes = (tf.TensorShape([None, self.p]),
                         tf.TensorShape([None, 1]))
        train_dataset = dd.from_generator(lambda: self.dtrain.one_iter(pheno_name),
                                          output_types=(tf.int8, tf.float16),
                                          output_shapes=output_shapes)
        dev_dataset = dd.from_generator(lambda: self.ddev.one_iter(pheno_name),
                                        output_types=(tf.int8, tf.float16),
                                        output_shapes=output_shapes)
        return train_dataset, dev_dataset

    def linear_model(self, epochs: int = 400, batch_size: int = 100,
                     l_rate: float = 0.001, penal: float = 0.005,
                     pheno_name: str = 'V1', tb_name: str = ''):
        """
        Linear Model

        :param epochs:
        :param batch_size:
        :param l_rate:
        :param penal:
        :param pheno_name:
        :param tb_name:
        :return: None
        """
        now = datetime.now()
        now = now.strftime('%d/%m/%Y-%H:%M:%S')
        lg.debug('Current time: %s', now)
        train_path = os.path.join(self.tb_path, 'train-'+tb_name+now)
        dev_path = os.path.join(self.tb_path, 'dev-'+tb_name+now)

        keep_prob = tf.placeholder(tf.float32, None, name='dropout_prob')
        train_dataset, dev_dataset = self._make_dataset(pheno_name)

        train_dataset = train_dataset.batch(batch_size)
        train_iter = train_dataset.make_initializable_iterator()
        dev_iter = dev_dataset.make_initializable_iterator()
        handle = tf.placeholder(tf.string, shape=[])
        iter = tf.data.Iterator.from_string_handle(handle,
                                                   train_dataset.output_types,
                                                   train_dataset.output_shapes)
        geno, pheno = iter.get_next()

        model = LinearModel(geno, pheno, self.bool_blocks,
                            keep_prob, l_rate, penal)

        train_writer = tf.summary.FileWriter(train_path,
                                             tf.get_default_graph())
        dev_writer = tf.summary.FileWriter(dev_path, tf.get_default_graph())
        merged_summary = tf.summary.merge_all()
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init)
            train_handle = sess.run(train_iter.string_handle())
            dev_handle = sess.run(dev_iter.string_handle())
            for i in range(epochs):
                sess.run(train_iter.initializer)
                while True:
                    try:
                        _, loss, summary = sess.run([model.optimize,
                                                     model.cost,
                                                     merged_summary],
                                                    feed_dict={keep_prob: 0.1,
                                                               handle: train_handle})
                    except tf.errors.OutOfRangeError:
                        break
                    train_writer.add_summary(summary, i)
                if i % 10 == 0:
                    sess.run(dev_iter.initializer)
                    while True:
                        try:
                            summary = sess.run([merged_summary],
                                               feed_dict={handle: dev_handle,
                                                          keep_prob: 1.0})
                            dev_writer.add_summary(summary, i)
                        except tf.errors.OutOfRangeError:
                            break
