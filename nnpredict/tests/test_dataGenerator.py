from unittest import TestCase
from nnpredict.kk_model import DataGenerator
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)


class TestDataGenerator(TestCase):

    def setUp(self):
        self.plink_file = 'data/sim_1000G_chr10'
        self.plink_file = 'data/sim_1000G_chr10'
        self.sample_major = 'data/sample_major/1kg/sim_1000G_chr10_SampleMajor_dev'
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
        sizes_blocks = keras_generator.dims
        sizes_x = [k.shape[1] for k in x]
        for k, l in enumerate(zip(sizes_x, sizes_blocks)):
            fx, fr = l
            self.assertEqual(fx, fr, "Found missmatch at "+str(k))
