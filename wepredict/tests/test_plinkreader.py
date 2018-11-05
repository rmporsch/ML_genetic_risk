"""Unittests for plink import"""
import unittest
from spyplink.plink_reader import Genetic_data_read
import logging as lg

lg.basicConfig(level=lg.DEBUG)


class Plink_Test(unittest.TestCase):
    """docstring for Plink_Test."""

    def setUp(self):
        self.plink_file = 'data/1kg_phase1_chr22'
        self.pheno_file = 'data/sim_1000G_chr10.txt'
        self.ld_block_file = 'data/Berisa.EUR.hg19.bed'

    def test_preprocessing(self):
        with self.assertRaises(ValueError):
            reader = Genetic_data_read(self.plink_file,
                                       self.ld_block_file, self.pheno_file)

    def test_ld_block(self):
        reader = Genetic_data_read(self.plink_file, self.ld_block_file)
        rs_id_group = reader.groups[22][0]
        len_rs_id_group = len(rs_id_group)
        genotypematrix = reader.block_read(22, rs_id_group)
        n, p = genotypematrix.shape
        self.assertEqual(n, 1092)
        self.assertEqual(p, len_rs_id_group)


if __name__ == '__main__':
    unittest.main()
