from unittest import TestCase
import os
import shutil
from spyplink.converting import Converting
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
lg = logging.getLogger(__name__)


class TestMajor_reader(TestCase):

    def setUp(self):
        self.OneKg = 'data/sim_1000G_chr10'
        self.pheno = 'data/sim_1000G_chr10.txt'
        self.output = 'data/test'
        self.dev = 'data/sample_major/1kg/sim_1000G_chr10_SampleMajor_dev.fam'
        self.train = 'data/sample_major/1kg/sim_1000G_chr10_SampleMajor_train.fam'
        os.mkdir(self.output)

    def tearDown(self):
        shutil.rmtree(self.output)
        try:
            os.remove('.snps.list')
            os.remove('.train.temp')
            os.remove('.dev.temp')
        except ValueError:
            print('something went wrong')

    def test_gwas(self):
        p = Converting(self.OneKg, self.output, self.pheno)
        train = pd.read_table(self.train, header=None)
        dev = pd.read_table(self.dev, header=None)
        p.add_train_dev_split(train, dev)
        args = ['--keep', '.train.temp']
        sumstat = p.run_gwas('V1', arguments=args)
        lg.debug('path to sumstat: %s', sumstat[0][1])
        ss = pd.read_table(sumstat[0][1])
        lg.debug('Number of rows %s and cols %s', *ss.shape)
        bim = pd.read_table(self.OneKg+'.bim', header=None)
        p, cols = bim.shape
        self.assertEqual(p, ss.shape[0])

    def test_clumping(self):
        p = Converting(self.OneKg, self.output, self.pheno)
        train = pd.read_table(self.train, header=None)
        dev = pd.read_table(self.dev, header=None)
        p.add_train_dev_split(train, dev)
        args = ['--keep', '.train.temp']
        sumstat = p.run_gwas('V1', arguments=args)
        clumped = p.run_clumping(sumstat, p1=0.01, p2=0.01, r2=0.1)
        lg.debug('List of clumped files %s', clumped)
        clumped_gwas = pd.read_table(clumped[0][1])
        n, pp = clumped_gwas.shape
        lg.debug('Number of rows %s and cols %s', *clumped_gwas.shape)
        self.assertEqual(pp, 13)
        self.assertEqual(n, 923)

    def test_converting(self):
        p = Converting(self.OneKg, self.output, self.pheno)
        train = pd.read_table(self.train, header=None)
        dev = pd.read_table(self.dev, header=None)
        p.add_train_dev_split(train, dev)
        args = ['--keep', '.train.temp']
        sumstat = p.run_gwas('V1', arguments=args)
        clumped = p.run_clumping(sumstat, p1=0.01, p2=0.01, r2=0.1)
        clumped_gwas = pd.read_table(clumped[0][1])
        snps = clumped_gwas.SNP
        snps.to_csv('.snps.list', index=False, header=None)

        add_arguments = ['--extract', '.snps.list']
        major_files = p.split_plink(add_arguments)

        lg.debug('Split output files %s', major_files)
        train_new = pd.read_table(major_files[0][0]+'.fam', header=None)
        n = train_new.shape[0]
        lg.debug('Loaded %s samples from training file', n)
        self.assertEqual(n, train.shape[0])
