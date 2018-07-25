import logging as lg
from dask.distributed import Client, LocalCluster
from wepredict.wepredict import wepredict
import unittest
import os
import numpy as np

lg.basicConfig(level=lg.DEBUG)

class Test_wepredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.plink_file = 'data/1kg_phase1_chr22'
        cls.pheno_file = 'data/simulated_chr10.txt'
        cls.ld_block_file = 'data/Berisa.EUR.hg19.bed'
        cls.monster = wepredict(cls.plink_file, cls.ld_block_file,
                                testing=True)
        

    def test_sim(self):
        X = self.monster.block_read(22, self.monster.blocks[22][0])
        y = self.monster._sim(X) self.assertEqual(len(y), 1092)

    def test_generation_of_training(self):
        sample = self.monster.generate_valid_test_data(0.1, 0.1)
        n = sum([len(sample[0]), len(sample[1]), len(sample[2])])
        self.assertEqual(n, 1092)


    def test_get_samples(self):
        X = self.monster.block_read(22, self.monster.blocks[22][0])
        n, p = X.shape
        y = self.monster._sim(X)
        index_train, index_valid, index_test = self.monster.generate_valid_test_data(0.1, 0.1)
        sample = self.monster.get_samples(X, y, index_train, index_valid)
        training_x = sample['training_x']
        self.assertEqual(training_x.shape[1], p)

    def test_subsamples(self):
        pheno = self.monster.fam
        n = 10
        ff = 'testing_pheno.txt'
        pheno = pheno.sample(n)
        pheno['P1'] = np.random.random(n)
        pheno.to_csv(ff, index=False, sep='\t')
        lg.debug('Header:\n %s', pheno.head())
        monster = wepredict(self.plink_file,
                            self.ld_block_file,
                            ff, testing=True)
        X = monster.block_read(22, monster.blocks[22][0])
        nx, px = X.shape
        self.assertEqual(n, nx)
        os.remove('testing_pheno.txt')

    def test_dask_simulation(self):
        cluster = LocalCluster()
        cluster.scale(3)
        client = Client(cluster)
        lg.info('%s', client)
        pheno = self.monster.simulate()
        self.assertFalse(pheno == None)
        lg.info('pheno shape is %s', pheno.shape)
        cluster.close()

