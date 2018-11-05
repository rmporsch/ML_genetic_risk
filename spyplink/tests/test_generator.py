from unittest import TestCase
import logging
import tensorflow as tf
from spyplink.plink_reader import get_genotypes
from nnpredict.nnpredict import NNpredict
import pandas as pd
import numpy as np

lg = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class TestMajor_reader(TestCase):

    def setUp(self):
        self.train = 'data/tf_keras_compare/sim_1000G_chr10_SampleMajor_train'
        self.dev = 'data/tf_keras_compare/sim_1000G_chr10_SampleMajor_dev'
        self.OneKg = 'data/sim_1000G_chr10'
        self.pheno = 'data/sim_1000G_chr10.txt'

    def test_datset_switch(self):
        batch_size = 50
        monster = NNpredict(self.train, self.dev, self.pheno,  '.')
        train_dataset, dev_dataset = monster._make_dataset('V1')
        train_dataset = train_dataset.batch(batch_size)
        dev_dataset = dev_dataset.batch(batch_size)

        train_iter = train_dataset.make_initializable_iterator()
        dev_iter = dev_dataset.make_initializable_iterator()
        handle = tf.placeholder(tf.string, shape=[], name='handle')
        iterr = tf.data.Iterator.from_string_handle(handle,
                                                    train_dataset.output_types)
        geno, pheno = iterr.get_next()
        bs = tf.shape(pheno, name='get_batchsize')[0]
        geno_r = tf.reshape(geno, (bs, monster.p), name='geno')
        pheno_r = tf.reshape(pheno, (bs, 1), name='pheno')

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        train_list = {'geno': [], 'pheno': []}
        dev_list = {'geno': [], 'pheno': []}

        with tf.Session() as sess:
            sess.run(init)
            train_handle = sess.run(train_iter.string_handle())
            dev_handle = sess.run(dev_iter.string_handle())
            train_dict = {handle: train_handle}
            dev_dict = {handle: dev_handle}
            lg.info('Reading Training Data')
            sess.run(train_iter.initializer)
            while True:
                try:
                    tgeno, tpheno = sess.run([geno_r, pheno_r],
                                             feed_dict=train_dict)
                    lg.debug('shape of tgeno %s', tgeno.shape)
                    train_list['geno'].append(tgeno)
                    train_list['pheno'].append(tpheno)
                except tf.errors.OutOfRangeError:
                    break
            lg.info('Reading Dev Data')
            sess.run(dev_iter.initializer)
            while True:
                try:
                    tgeno, tpheno = sess.run([geno_r, pheno_r],
                                             feed_dict=dev_dict)
                    dev_list['geno'].append(tgeno)
                    dev_list['pheno'].append(tpheno)
                except tf.errors.OutOfRangeError:
                    break

        # read data to compare input
        lg.info('Finished with tensorflow')
        bim = pd.read_table(self.dev + '.bim', header=None)
        snps = bim.iloc[:, 1].values
        dev_fam = pd.read_table(self.dev + '.fam', header=None)
        dev_n = dev_fam.shape[0]
        train_fam = pd.read_table(self.train + '.fam', header=None)
        train_n = train_fam.shape[0]
        plink_fam = pd.read_table(self.OneKg + '.fam', header=None, sep=' ')
        lg.info('Loaded information to import from the original data')

        train_id = plink_fam.iloc[:, 1].isin(train_fam.iloc[:, 1]).values
        assert sum(train_id) == train_n
        dev_id = plink_fam.iloc[:, 1].isin(dev_fam.iloc[:, 1]).values
        assert sum(dev_id) == dev_n

        var_train_genotypes = get_genotypes(snps, self.OneKg, train_id)
        var_dev_genotypes = get_genotypes(snps, self.OneKg, dev_id)
        lg.info('Loaded genotype from variant major format')

        sample_train_genotypes = np.concatenate(train_list['geno'], axis=0)
        sample_dev_genotypes = np.concatenate(dev_list['geno'], axis=0)
        lg.info('Stacked genotypes from tf')

        lg.debug('Sample Train Shape: %s with sum of %s',
                 sample_train_genotypes.shape, np.sum(sample_train_genotypes))
        lg.debug('Sample Dev Shape: %s with sum of %s',
                 sample_dev_genotypes.shape, np.sum(sample_dev_genotypes))

        lg.debug('Var Train Shape: %s with sum of %s',
                 var_train_genotypes.shape, np.sum(var_train_genotypes))
        lg.debug('Var Dev Shape: %s with sum of %s',
                 var_dev_genotypes.shape, np.sum(var_dev_genotypes))

        lg.debug('unique elements in var train %s',
                 np.unique(var_train_genotypes))


        self.assertTrue(np.array_equal(var_train_genotypes,
                                       sample_train_genotypes))
        self.assertTrue(np.array_equal(var_dev_genotypes,
                                       sample_dev_genotypes))

        phenotype = pd.read_table(self.pheno)
        pheno = phenotype.V1.values
        var_train_pheno = pheno[train_id]
        var_dev_pheno = pheno[dev_id]

        flatten = lambda l: [item for sublist in l for item in sublist]

        sample_train_pheno = np.array(flatten(train_list['pheno']))
        sample_dev_pheno = np.array(flatten(dev_list['pheno']))
        lg.info('Got phenotypes from tf')
        lg.debug('Var train with shape %s and sum %s',
                 sample_train_pheno.shape, np.sum(sample_train_pheno))
        lg.debug('Var dev with shape %s and sum %s',
                 sample_dev_pheno.shape, np.sum(sample_dev_pheno))

        lg.debug('Var train with shape %s and sum %s',
                 var_train_pheno.shape, np.sum(var_train_pheno))
        lg.debug('Var dev with shape %s and sum %s',
                 var_dev_pheno.shape, np.sum(var_dev_pheno))

