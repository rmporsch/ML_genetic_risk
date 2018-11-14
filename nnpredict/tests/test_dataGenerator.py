from unittest import TestCase
from nnpredict.kk_model import DataGenerator
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)


class TestDataGenerator(TestCase):

    def setUp(self):
        self.plink_file = 'data/sim_1000G_chr10'
        self.sample_major = 'data/test'
        self.pheno_file = 'data/sim_1000G_chr10.txt'
        self.ld_block_file = 'data/Berisa.EUR.hg19.bed'
        self.sample_major_numpy = 'data/sample_major_1kg/sample_major_0.npy'

    def test__data_generation(self):
        keras_generator = DataGenerator(self.sample_major,
                                        self.pheno_file,
                                        'V1', 100, shuffle=False)
        pheno = pd.read_table(self.pheno_file)
        pheno = pheno.V1.values
        mat = np.load(self.sample_major_numpy)[0]
        n, p = mat.shape
        index = np.arange(0, n)
        x, y = keras_generator._data_generation(index, 'V1')
        is_equal_geno = np.equal(x, mat).all()
        self.assertTrue(is_equal_geno)
        is_equal_pheno = np.equal(y, pheno[index]).all()
        self.assertTrue(is_equal_pheno)

    def test_get_item(self):
        keras_generator = DataGenerator(self.sample_major,
                                        self.pheno_file,
                                        'V1', 100, shuffle=False)
        pheno = pd.read_table(self.pheno_file)
        pheno = pheno.V1.values
        mat = np.load(self.sample_major_numpy)[0]
        n, p = mat.shape
        index = np.arange(0, n)
        x, y = keras_generator.__getitem__(0)
        is_equal_geno = np.equal(x, mat).all()
        self.assertTrue(is_equal_geno)
        is_equal_pheno = np.equal(y, pheno[index]).all()
        self.assertTrue(is_equal_pheno)

    def test_split_generation(self):
        keras_generator = DataGenerator(self.sample_major,
                                        self.pheno_file,
                                        'V1', 100, shuffle=False,
                                        ldblock_file=self.ld_block_file)

        x, y = keras_generator.__getitem__(0)
        self.assertEqual(len(x), 85)
        nrows = list()
        for i in x:
            nrows.append(i.shape[0])
        num_unique = len(np.unique(nrows))
        self.assertEqual(1, num_unique)
