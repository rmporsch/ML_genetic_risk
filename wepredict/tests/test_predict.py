from unittest import TestCase
from wepredict.predict import Predict
import logging as lg
import pickle

lg.basicConfig(level=lg.DEBUG)

class TestPredict(TestCase):

    def setUp(self):
        self.plink_dev_file = 'data/test_data/test_split.dev'
        self.plink_train_file = 'data/test_data/test_split.train'
        self.pheno_file = 'data/sim_1000G_chr10.txt'
        self.ld_block_file = 'data/Berisa.EUR.hg19.bed'

    def test_fit(self):
        model = Predict(self.plink_train_file, self.plink_dev_file,
                        self.pheno_file, 100)
        model.fit('V1', 'l1', 0.01, 0.001, 50)
        pickle.dump(model.results,
                    open('data/test_data/test_output.pickle', 'wb'))
