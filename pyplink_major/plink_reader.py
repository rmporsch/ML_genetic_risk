"""Read plink files."""
from pyplink import PyPlink
import pandas as pd
import os
import numpy as np
import logging
import pickle
from bitarray import bitarray
from typing import List
from typing import Tuple

BoolVector = List[bool]
SetTuple = Tuple[str, str]

lg = logging.getLogger(__name__)

def get_genotypes(rsid, plink_path, sub_in):
    """
    Retrive genotype matrix from variant major format

    :param rsid: list of rsids
    :param plink_path: plink-stem path
    :param sub_in: list of subjects to inlucde
    :return: genotypematrix
    """
    reader = PyPlink(plink_path)
    n = reader.get_nb_samples()
    genotypematrix = np.zeros((n, len(rsid)), dtype=np.int8)
    pos_index = 0
    for snp, genotype in reader.iter_geno_marker(rsid):
        if snp not in rsid:
            continue
        else:
            genotypematrix[:, pos_index] = genotype[sub_in]
            pos_index += 1
    reader.close()
    return genotypematrix


class Major_reader(object):

    def __init__(self, plink_file: str, pheno: str = None,
                 ldblock_file: str = None):
        """
        Reading plink files in sample major format.

        :param plink_file: plink file stem
        :param pheno: path to pheno file
        :param ldblock_file: path to ld block file
        """
        super(Major_reader, self).__init__()
        self.plink_file = plink_file
        self.pheno_file = pheno
        self.pheno = self._read_pheno(pheno)
        self.bim = pd.read_table(plink_file+'.bim', header=None,
                                 names=['chr', 'rsid', 'c', 'pos',
                                        'a1', 'a2'])
        self.p = self.bim.shape[0]
        self.chr = self.bim.chr.unique()
        lg.debug('Using %s number of SNPs', self.p)
        assert os.path.isfile(self.pheno_file)
        assert os.path.isfile(self.plink_file+'.bed')
        self._is_sample_major = self._check_magic_number(self.plink_file)
        self._size = -(-self.p // 4)
        self._overflow = self._size*4 - self.p
        self._to_remove = [self.p + k for k in range(self._overflow)]
        lg.debug('Removing bytes: %s', self._to_remove)
        if ldblock_file is not None:
            self.ldblocks = self._check_ldblocks(ldblock_file)
        else:
            self.ldblocks = None

    def _check_ldblocks(self, ldblocks_path: str):
        pickel_path = self.plink_file+'.ld_blocks.pickel'
        if os.path.isfile(pickel_path):
            with open(pickel_path, 'rb') as f:
                blocks = pickle.load(f)
        else:
            blocks = pd.read_csv(ldblocks_path, sep='\t')
            blocks.columns = [k.strip() for k in blocks.columns]
            blocks['chr'] = blocks['chr'].str.strip('chr')
            blocks['chr'] = blocks['chr'].astype(int)
            blocks = self._preprocessing_ldblock(blocks)
            pickle.dump(blocks, open(pickel_path, 'wb'))
        return blocks

    def _preprocessing_ldblock(self, blocks) -> dict:
        out = {}
        for chr in self.chr:
            subset_blocks = blocks[blocks.chr == chr]
            subset_bim = self.bim[self.bim.chrom == chr]
            out[chr] = []
            for index, row in subset_blocks.iterrows():
                start = row['start']
                end = row['stop']
                rsids = subset_bim[
                    (self.bim.pos >= start)
                    & (self.bim.pos <= end)
                    ].index.values
                out[chr].append(rsids)
        return out

    def _read_pheno(self, pheno: str = None):
        columns = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO']
        fam = pd.read_table(self.plink_file+'.fam', header=None)
        self.n = fam.shape[0]
        if fam.shape[1] > 5:
            fam.columns = columns
        elif fam.shape[1] == 5:
            fam.columns = columns[:-1]
        else:
            raise ValueError('The fam file seems wrongly formated')
        fam[['FID', 'IID']] = fam[['FID', 'IID']].applymap(str)

        if pheno is not None:
            pheno = pd.read_table(pheno)
            pheno[['FID', 'IID']] = pheno[['FID', 'IID']].applymap(str)
            n_import = pheno.shape[0]
            header = pheno.columns
            if np.all([k not in ['FID', 'IID'] for k in header]):
                raise ValueError('FID and IID needs to be a pheno file')
            if np.all([k in header for k in ['PAT', 'MAT', 'SEX']]):
                pheno.drop(['PAT', 'MAT', 'SEX'], axis=1, inplace=True)

            mpheno = pd.merge(fam, pheno, 'left', on=['IID', 'FID'])
            merged_n, merged_p = mpheno.shape
            lg.debug('Format of the new fam file %s', mpheno.shape)
            if merged_n < n_import:
                lg.warning('Out of %s subjects, %s were in fam file',
                           n_import, merged_n)
                if merged_n == 0:
                    raise ValueError('No subject present in file')
            pheno_columns = [k for k in mpheno.columns if k not in fam.columns]
            lg.info('Extracted and integrated the following phenotypes %s',
                    pheno_columns)
            for p in pheno_columns:
                missing = mpheno[p].isnull().mean()
                lg.debug('Missingness in %s is %s', p, missing)
                if missing > 0.8:
                    lg.warning('Removed %s due to missingnesses (>80%)')
                    mpheno.drop(p, axis=1)
            self.pheno_names = pheno_columns
            self.subject_ids = mpheno.IID.values
        else:
            mpheno = fam
            if mpheno.shape[1] < 6:
                raise ValueError('No phenotype present')
        return mpheno

    def _check_magic_number(self, plink_file: str):
        variant_major = '6c1b01'
        sample_major = '6c1b00'
        with open(plink_file+'.bed', 'rb') as f:
            magic = f.read(3)
            magic = magic.hex()
        lg.debug('first three bytes in hex: %s', magic)
        if magic == variant_major:
            raise ValueError('Please use bed file in sample-major format')
        elif magic == sample_major:
            lg.info('sample major format')
            return True
        else:
            raise ValueError('Could not identify plink file')

    @staticmethod
    def _bgeno(input_bytes: bytes):
        a = bitarray(endian='little')
        a.frombytes(input_bytes)
        # lg.debug('How does the bytes look like: %s', a)
        # encoding missing 01 as 0
        d = {2: bitarray(b'00'), 1: bitarray(b'01'),
             0: bitarray(b'11'), 9: bitarray(b'10')}
        # lg.debug("Decoding dict: %s", d)
        genotypes = a.decode(d)
        return genotypes

    def _binary_genotype(self, input_bytes: bytes, snps: BoolVector = None):
        genotypes = self._bgeno(input_bytes)
        if snps is not None:
            p = len(snps)
        else:
            p = self.p
        for k in sorted(self._to_remove, reverse=True):
            del genotypes[k]
        genotypes = np.array(genotypes, dtype=np.float16)
        if snps is not None:
            genotypes = genotypes[np.array(snps)]
        genotypes[genotypes == 9] = 0
        return genotypes.reshape(1, p)

    def _iter_geno(self, mini_batch_size: int, snps: BoolVector = None):
        if snps is None:
            p = self.p
        else:
            assert self.p == len(snps)
            p = sum(snps)

        with open(self.plink_file+'.bed', 'rb') as f:
            input_bytes = f.read(3)
            while True:
                genotype_matrix = np.zeros((mini_batch_size, p))
                for sample in range(mini_batch_size):
                    input_bytes = f.read(self._size)
                    if not input_bytes:
                        f.seek(3)
                        input_bytes = f.read(self._size)
                    genotype_matrix[sample, :] = self._binary_genotype(input_bytes,
                                                                       snps)
                yield genotype_matrix

    def _one_iter_geno(self, snps: list = None):
        if snps is not None:
            assert self.p == len(snps)
        with open(self.plink_file+'.bed', 'rb') as f:
            input_bytes = f.read(3)
            while input_bytes != '':
                input_bytes = f.read(self._size)
                if input_bytes:
                    yield self._binary_genotype(input_bytes, snps)
                else:
                    break

    def _one_iter_pheno(self, pheno):
        y = self.pheno[pheno]
        for i in y:
            yield np.array([i]).reshape(1, 1)

    def one_iter(self, pheno: str, snps: list = None):
        """
        Simple interator for one single sample at a time

        :param pheno: phenotype to iterate over
        :param snps: potential subset of SNPs (optional)
        :return: (geno, pheno)
        """
        pheno_iter = self._one_iter_pheno(pheno)
        geno_iter = self._one_iter_geno(snps)

        for geno, pheno in zip(geno_iter, pheno_iter):
            if np.isnan(pheno):
                lg.debug('Found missing value, skipping')
                continue
            else:
                yield geno, pheno


    def _iter_pheno(self, pheno: str, mini_batch_size: int):
        start = 0
        end = mini_batch_size
        index = np.arange(self.n)
        iter_over_all = False
        y = self.pheno[pheno]
        while True:
            if end >= self.n:
                end = end - self.n
            if start >= self.n:
                start = start - self.n
            bool_index = ~((index >= start) ^ (index < end))
            if start > end:
                bool_index = ~(bool_index)
                iter_over_all = True
            batch_index = index[bool_index]
            yyield = np.zeros((mini_batch_size, 1))
            for u, i in np.ndenumerate(batch_index):
                yyield[u] = y[i]
            start = np.abs(end)
            end = start + mini_batch_size
            yield yyield

    def read(self, phenotype: str, mini_batch_size: int = 1,
             snps: BoolVector = None):
        """
        Reader for plink major file format.

        :param phenotype: str of phenotype name
        :param mini_batch_size: how many samples to load
        :param snps: an optional list of bool to inlcude/exclude certain snps
        :return: genotype_matrix
        """
        pheno_iter = self._iter_pheno(phenotype, mini_batch_size)
        geno_iter = self._iter_geno(mini_batch_size, snps)

        for geno, pheno in zip(geno_iter, pheno_iter):
            lg.debug('Shape of geno: %s Shape of pheno: %s', geno.shape,
                     pheno.shape)
            yield geno, pheno


