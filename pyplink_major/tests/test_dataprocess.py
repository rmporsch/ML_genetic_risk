from unittest import TestCase
import logging
from pyplink_major.data_processing import PreProcess
import pandas as pd
import numpy as np

lg = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class TestMajor_reader(TestCase):

    def setUp(self):
        self.pfile = 'data/sim_1000G_chr10'

    def test_counter(self):
        proc = PreProcess(self.pfile)
        bim = pd.read_table(self.pfile+'.bim')
        num_snps = bim.shape[0]
        simple_read = proc._file_len(self.pfile+'.bim')
        self.assertEqual(num_snps, simple_read)

    def test_count_by_n(self):
        proc = PreProcess(self.pfile)
        fam = pd.read_table(self.pfile+'.fam')
        bim = pd.read_table(self.pfile+'.bim')
        num_snps = bim.shape[0]
        n = fam.shape[0]
        seq = np.arange(100, n, 200, dtype=int)
        seq = np.append(seq, n)
        lg.debug('Using sequence %s', seq)
        counter = proc.count_variants('.', seq, self.pfile)
        self.assertEqual(counter[str(seq[-1])], num_snps)

