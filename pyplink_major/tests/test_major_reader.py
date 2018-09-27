from unittest import TestCase
from pyplink_major.plink_reader import Major_reader
from pyplink_major.plink_reader import split_plink
from bitarray import bitarray
import logging
import numpy as np
import pandas as pd

lg = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class TestMajor_reader(TestCase):

    def setUp(self):
        self.plink_file = 'data/sim_1000G_chr10'
        self.sample_major = 'data/test'
        self.pheno_file = 'data/sim_1000G_chr10.txt'
        self.ld_block_file = 'data/Berisa.EUR.hg19'
        self.sample_major_numpy = 'data/sample_major_1kg/sample_major_0.npy'

    def test_split_plink(self):
        ratio = 0.8
        outptut = 'data/test_data/test_split'
        train, dev = split_plink(self.plink_file, outptut, ratio, 100)
        lg.debug('Train: %s, Dev: %s', train, dev)
        train_fam = pd.read_table(train+'.fam', header=None)
        train_n = train_fam.shape[0]
        dev_fam = pd.read_table(dev+'.fam', header=None)
        dev_n = dev_fam.shape[0]
        fam = pd.read_table(self.plink_file+'.fam', header=None)
        n = fam.shape[0]
        expected_n_train = [1900, 2000]
        expected_n_dev =[400, 500]
        self.assertIn(train_n, expected_n_train)
        self.assertIn(dev_n, expected_n_dev)


    def test_check_magic_number(self):
        with self.assertRaises(ValueError):
            out = Major_reader(self.plink_file, self.pheno_file)
        out = Major_reader(self.sample_major, self.pheno_file)
        self.assertTrue(out._is_sample_major)

    def test_binary_genotype(self):
        bits = '00011011'
        expected_genotypes = [0, 1, 9, 2]
        a = bitarray(bits)
        input_bytes = a.tobytes()
        lg.debug('Used bytes: %s', input_bytes)
        out = Major_reader(self.sample_major, self.pheno_file)
        genotypes = out._bgeno(input_bytes)
        lg.debug('outputed genotypes: %s', genotypes)
        comparision = [genotypes[i] == expected_genotypes[i] for i in range(4)]
        lg.debug('Comparision result: %s', comparision)
        comparision = sum(comparision)
        self.assertEqual(comparision, 4)

    def test_binary_genotype_overflow(self):
        expected_genotypes = [0, 1, 0, 2, 2, 2]
        a = bitarray('00011011' '11110000', endian='big')
        size = -(-len(expected_genotypes) // 4)
        over_flow = size*4 - len(expected_genotypes)
        to_remove = [len(expected_genotypes) + k for k in range(over_flow)]
        input_bytes = a.tobytes()
        lg.debug('Used bytes: %s', input_bytes)
        out = Major_reader(self.sample_major, self.pheno_file)
        out._to_remove = to_remove
        lg.debug('Removing the following: %s', to_remove)
        genotypes = out._binary_genotype(input_bytes)
        self.assertEqual(len(genotypes), len(expected_genotypes))
        lg.debug('outputed genotypes: %s', genotypes)
        comparision = [genotypes[i] == expected_genotypes[i] for i in range(6)]
        lg.debug('Comparision result: %s', comparision)
        comparision = sum(comparision)
        self.assertEqual(comparision, 6)

    def test_geno_read(self):
        gold_data = np.load(self.sample_major_numpy)
        lg.debug('index gold: %s', gold_data[1][0:10])
        gold_data = gold_data[0]
        n_gold, p_gold = gold_data.shape
        lg.debug('Number of samples: %s Number of SNPs %s in gold',
                 n_gold, p_gold)
        out = Major_reader(self.sample_major, self.pheno_file)
        reader = out._iter_geno(n_gold)
        genotype_matrix = next(reader)
        n, p = genotype_matrix.shape
        self.assertEqual(n_gold, n)
        self.assertEqual(p_gold, p)
        lg.debug('Gold: %s', gold_data[0, 0:10])
        lg.debug('Sample-Major: %s', genotype_matrix[0, 0:10])
        sub_i = genotype_matrix[0] == gold_data[0]
        self.assertEqual(np.sum(sub_i), p)

    def test_continious_geno_read(self):
        fam = pd.read_table(self.pheno_file)
        batch_size = 100
        nn = fam.shape[0]
        out = Major_reader(self.sample_major, self.pheno_file)
        reader = out._iter_geno(100)
        first = next(reader)

        to_end = nn // batch_size - 1
        overlap = nn - nn // batch_size
        lg.debug('estimated overlap is %s', overlap)
        lg.debug('steps to end: %s', to_end)
        for i in range(to_end):
            lg.debug(i)
            batch = next(reader)
        batch = next(reader)
        lg.debug('shape of first is %s', first.shape)
        lg.debug('shape of last is %s', batch.shape)
        expected_overlap = batch_size*(nn // batch_size +1) - nn
        compare = batch[4] == first[0]
        lg.debug(compare)
        self.assertEqual(out.p, np.sum(compare))

    def test_pheno_reader(self):
        pheno = pd.read_table(self.pheno_file)
        batch_size = 100
        out = Major_reader(self.sample_major, self.pheno_file)
        reader = out._iter_pheno('V1', batch_size)
        batch = next(reader)
        compare = batch == pheno.V1.values[:batch_size]
        lg.debug(compare[0:10])
        self.assertTrue(batch_size, np.sum(compare))

    def test_continious_pheno_reader(self):
        fam = pd.read_table(self.pheno_file)
        batch_size = 100
        nn = fam.shape[0]
        out = Major_reader(self.sample_major, self.pheno_file)
        reader = out._iter_pheno('V1', 100)
        first = next(reader)

        to_end = nn // batch_size - 1
        overlap = nn - nn // batch_size
        lg.debug('estimated overlap is %s', overlap)
        lg.debug('steps to end: %s', to_end)
        for i in range(to_end):
            lg.debug(i)
            batch = next(reader)
        batch = next(reader)
        first = first[-overlap:, :]
        lg.debug('shape of first is %s', first.shape)
        batch = batch[:overlap, :]
        lg.debug('shape of last is %s', batch.shape)
        compare = batch == first
        lg.debug(compare[0:10])
        expected_overlap = batch_size*(nn // batch_size +1) - nn
        self.assertEqual(expected_overlap, np.sum(compare))

    def test_double_iter(self):
        out = Major_reader(self.sample_major, self.pheno_file)
        reader = out.read('V1', 100)
        geno, pheno = next(reader)
        n_geno = geno.shape[0]
        n_pheno = len(pheno)
        self.assertEqual(n_geno, n_pheno)

