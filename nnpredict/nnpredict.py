import os
import numpy as np
import logging
import pickle
import tensorflow as tf
from nnpredict.models import LinearModel, NNModel
from datetime import datetime
from pyplink_major import plink_reader as pr
import shutil

lg = logging.getLogger(__name__)

class NNpredict(object):
    """Neural Network prediction."""

    def __init__(self, plink_file_train: str, plink_file_dev: str,
                 pheno_file: str,
                 tb_path: str = 'tensorboard/neural_network/'):
        super(NNpredict, self).__init__()
        self.dtrain = pr.Major_reader(plink_file_train, pheno_file)
        self.ddev = pr.Major_reader(plink_file_dev, pheno_file)
        assert self.dtrain.p == self.ddev.p
        self.p = self.ddev.p
        self.n = self.dtrain.n
        self.tb_path = tb_path


    def _make_dataset(self, pheno_name: str):
        dd = tf.data.Dataset()
        output_shapes = (tf.TensorShape([None, self.p]),
                         tf.TensorShape([None, 1]))
        train_dataset = dd.from_generator(lambda: self.dtrain.one_iter(pheno_name, shuffle=True),
                                          output_shapes=output_shapes,
                                          output_types=(tf.float32, tf.float32))
        dev_dataset = dd.from_generator(lambda: self.ddev.one_iter(pheno_name, shuffle=True),
                                        output_shapes=output_shapes,
                                        output_types=(tf.float32, tf.float32))
        return train_dataset, dev_dataset

    def _make_tb_paths(self, penal, l_rate, batch_size, tb_name):
        """
        Makes a name and path for the tensorboard

        :param penal: penalty value
        :param l_rate: learning rate
        :param batch_size: batchsize
        :param tb_name: tbname
        :return: path to train and dev tensorboard
        """

        now = datetime.now()
        now = now.strftime('%Y-%m-%d-%H:%M:%S')
        param = 'lambda:%f:l_rate:%f:batchsize:%i' % (penal, l_rate,
                                                      batch_size)
        lg.debug('Current time: %s', now)
        train_path = os.path.join(self.tb_path, 'train-'+tb_name+param+now)
        dev_path = os.path.join(self.tb_path, 'dev-'+tb_name+param+now)
        lg.debug('Saving tensorboard at:\n train: %s \n test: %s',
                 train_path, dev_path)
        return train_path, dev_path

    @staticmethod
    def _make_output_path(export_dir, pheno_name):
        """
        Make output path

        :param export_dir: target path
        :param pheno_name: pheno name
        :return: string with path
        """
        export_dir = os.path.join(export_dir, 'tf_model_'+pheno_name)
        lg.info('Writing finished model to %s', export_dir)
        if os.path.isdir(export_dir):
            lg.info('output dir already exists, deleting')
            shutil.rmtree(export_dir)
        return export_dir

    def run_model(self, epochs: int = 400, batch_size: int = 100,
                     l_rate: float = 0.001, penal: float = 0.005,
                     pheno_name: str = 'V1', tb_name: str = '',
                  in_model=LinearModel, export_dir: str = os.getcwd(),
                  keep: float = 0.2, **kwargs):
        """
        Run Tensorflow model.

        :param epochs:
        :param batch_size:
        :param l_rate:
        :param penal:
        :param pheno_name:
        :param tb_name:
        :param in_model: Class of LinearModel or NN
        :param export_dir:
        :param keep: percentage of neurons to keep during training
        :return: None
        """
        assert ((keep <= 1.0) and (keep >= 0.0))
        export_dir = self._make_output_path(export_dir, pheno_name)
        tb_train_path, tb_dev_path = self._make_tb_paths(penal, l_rate,
                                                         batch_size, tb_name)
        train_dataset, dev_dataset = self._make_dataset(pheno_name)

        train_dataset = train_dataset.batch(batch_size)
        dev_dataset = dev_dataset.batch(50)

        train_iter = train_dataset.make_initializable_iterator()
        dev_iter = dev_dataset.make_initializable_iterator()
        handle = tf.placeholder(tf.string, shape=[], name='handle')
        iterr = tf.data.Iterator.from_string_handle(handle,
                                                    train_dataset.output_types)
        geno, pheno = iterr.get_next()
        bs = tf.shape(pheno, name='get_batchsize')[0]
        geno_r = tf.reshape(geno, (bs, self.p), name='geno')
        pheno_r = tf.reshape(pheno, (bs, 1), name='pheno')
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')

        lg.debug('Type of geno_sq: %s ', geno.dtype)
        lg.debug('Type of pheno_sq: %s ', pheno.dtype)

        model = in_model(geno_r, pheno_r, keep_prob, l_rate, penal, **kwargs)

        train_writer = tf.summary.FileWriter(tb_train_path,
                                             tf.get_default_graph())
        dev_writer = tf.summary.FileWriter(tb_dev_path, tf.get_default_graph())

        merged_summary = tf.summary.merge_all()
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        lg.debug('Completed model build and collected variables.')
        with tf.Session() as sess:
            sess.run(init)
            train_handle = sess.run(train_iter.string_handle())
            dev_handle = sess.run(dev_iter.string_handle())
            train_dict = {keep_prob: keep, handle: train_handle}
            dev_dict = {keep_prob: 1.0, handle: dev_handle}
            lg.debug('Starting epochs.')
            for i in range(epochs):
                sess.run(train_iter.initializer)
                sess.run(tf.local_variables_initializer())
                lg.info('Epoch %s', i)
                while True:
                    try:
                        _, loss, summary, er = sess.run([model.optimize,
                                                         model.cost,
                                                         merged_summary,
                                                         model.error],
                                                        feed_dict=train_dict)
                        lg.debug('Loss %s, Error %s', loss, er)
                    except tf.errors.OutOfRangeError:
                        break
                train_writer.add_summary(summary, i)
                train_writer.flush()
                lg.info('TRAIN: Epoch: %s: Loss %s, Error %s', i, loss, er)
                if i % 10 == 0:
                    lg.debug('Finished epoch %s running dev set', i)
                    sess.run(dev_iter.initializer)
                    sess.run(tf.local_variables_initializer())
                    testing = []
                    while True:
                        try:
                            summary, loss, er, y_pred, yy = sess.run([merged_summary,
                                                                      model.cost,
                                                                      model.error,
                                                                      model.prediction,
                                                                      pheno_r],
                                                                     feed_dict=dev_dict)
                            corr = np.corrcoef(y_pred.flatten(), yy.flatten())
                            testing.append(corr[0, 1])
                            lg.debug('Finished minibatch - dev Err: %s', er)
                        except tf.errors.OutOfRangeError:
                            break
                    lg.debug('MANUAL CORRELATION: %s', np.mean(testing))
                    dev_writer.add_summary(summary, i)
                    dev_writer.flush()
                    lg.info('DEV: Epoch: %s: Loss %s, Error %s', i, loss, er)
            # save model to disk
            tf.saved_model.simple_save(sess, export_dir, {'geno': geno_r},
                                       {'pheno': pheno_r})
            return er

