import logging as lg
import unittest
from wepredict.wepredict import wepredict
from wepredict.pytorch_regression import pytorch_linear
from sklearn.preprocessing import scale
import torch
import numpy as np

lg.basicConfig(level=lg.DEBUG)

class Test_Pytorch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        plink_file = 'data/1kg_phase1_chr22'
        pheno_file = 'data/simulated_chr10.txt'
        ld_block_file = 'data/Berisa.EUR.hg19.bed'
        cls.monster = wepredict(plink_file, ld_block_file, testing=False)
        cls.mini_batch = 50
        cls.X = cls.monster.block_read(22, cls.monster.blocks[22][0])
        cls.y = cls.monster._sim(cls.X)
        cls.indexes = cls.monster.generate_valid_test_data(0.1, 0.1)
        cls.sample = cls.monster.get_samples(cls.X, cls.y, cls.indexes[0],
                                             cls.indexes[1])
        cls.torch_reg = pytorch_linear(cls.sample['training_x'],
                                       cls.sample['training_y'],
                                       cls.sample['valid_x'],
                                       cls.sample['valid_y'], type='c',
                                       mini_batch_size=cls.mini_batch,
                                       if_shuffle=False)

    def test_loss(self):
        y_hat = np.array([0, 0])
        y = np.array([0, 0])
        loss = self.torch_reg._loss_function(y_hat, y)
        self.assertEqual(loss, 0)

        y_hat = np.array([0.1, 0.1])
        y = np.array([0.2, 0.1])
        loss = self.torch_reg._loss_function(y_hat, y)
        self.assertAlmostEqual(loss, 0.005)

    def test_accu(self):
        predict = torch.from_numpy(self.sample['valid_y'])
        pred = self.torch_reg._accuracy(predict)
        self.assertEqual(1.0, pred)

    def test_iterator(self):
        iter = self.torch_reg.iterator()
        xyield, yyield = next(iter)
        real_x = scale(self.sample['training_x'][0:50, :])
        n, p = real_x.shape
        n_yield, p_yield = xyield.shape
        self.assertEqual(n, n_yield)
        self.assertEqual(p, p_yield)
        overlap = real_x[0] == xyield[0]
        overlap = np.mean(overlap)
        self.assertEqual(overlap, 1.0)

    def test_run(self):
        self.torch_reg.mini_batch_size = 400
        output = self.torch_reg.run(penal='l1', epochs=105)
        self.assertEqual(output.type, 'c')
        self.assertEqual(len(output.iter_accu), 1)
        self.assertLess(0.5, output.accu)

        output = self.torch_reg.run(penal='l0', epochs=105)
        self.assertEqual(output.type, 'c')
        self.assertEqual(len(output.iter_accu), 1)
        self.assertLess(0.5, output.accu)
        self.assertEqual(len(output.coef), 3)
        import pickle
        with open('need_to_check.pickle', 'wb') as f:
            pickle.dump(self.torch_reg.saver_check, f)

