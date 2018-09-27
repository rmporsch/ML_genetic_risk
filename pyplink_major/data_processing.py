"""
Class to prep data
"""
import os
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple
import subprocess
from glob import glob
import filecmp

lg = logging.getLogger(__name__)
BoolVector = List[bool]
SetTuple = Tuple[str, str]


class PreProcess(object):

    def __init__(self, plink_path: str, ld_block_path: str = None):
        super(PreProcess, self).__init__()
        self.plink_path = plink_path
        self.ld_block_path = ld_block_path
        self.plink_files = self._expand_path(plink_path)
        self.sample_major = self._check_files(self.plink_files)
        self.plink2_binary =  os.path.join(os.path.dirname(__file__),
                                           'bin/plink2')
        self.plink_binary =  os.path.join(os.path.dirname(__file__),
                                           'bin/plink')
        lg.debug('location of plink2 file: %s', self.plink2_binary)
        lg.debug('location of plink file: %s', self.plink_binary)
        if ld_block_path is not None:
            self.ldblocks = self._load_ldblockfile(ld_block_path)
        else:
            self.ldblocks = None

    @staticmethod
    def _expand_path(plink_path: str):
        """
        Expands path to multiple plink files
        """
        if '*' in plink_path:
            files = glob(plink_path)
            bed_files = [x for x in files if x.endswith('bed')]
            plink_stemfiles = [os.path.splitext(x)[0] for x in bed_files]
            lg.debug('Selected plink files %s', plink_stemfiles)
            return plink_stemfiles
        else:
            lg.debug('Selected plink files %s', [plink_path])
            return [plink_path]

    def _check_files(self, plink_path: List):
        plink_type = list()
        fam1 = plink_path[0]+'.fam'
        for p in plink_path:
            lg.debug('Checking %s', p)
            assert os.path.isfile(p+'.bim')
            assert os.path.isfile(p+'.fam')
            assert filecmp.cmp(fam1, p+'.fam')
            assert os.path.isfile(p+'.bed')
            plink_type.append(self._check_plink_type(p))
        if len(set(plink_type)) == 1:
            return plink_type[0]
        else:
            raise ValueError('Not all bed files are of the same type.')

    @staticmethod
    def _check_plink_type(plink_path: str):
        variant_major = '6c1b01'
        sample_major = '6c1b00'
        with open(plink_path+'.bed', 'rb') as f:
            magic = f.read(3)
            magic = magic.hex()
        lg.debug('first three bytes in hex: %s', magic)
        if magic == variant_major:
            lg.info('bed file in variant major format.')
            return False
        elif magic == sample_major:
            lg.info('bed file in sample major format.')
            return True
        else:
            raise ValueError('Could not recognice bed format.')

    @staticmethod
    def _load_bim(bimfile):
        headernames = ['chr', 'rsid', 'm', 'bp', 'a1', 'a2']
        bim = pd.read_table(bimfile, header=None, names=headernames)
        lg.debug('dtype of chr in bim file is %s', bim.chr.dtype)
        if (bim.chr.dtype == np.float) or (bim.chr.dtype == np.int):
            return bim
        else:
            bim['chr'] = bim['chr'].str.strip('chr')
            bim['chr'] = bim['chr'].astype(int)
            return bim

    @staticmethod
    def _load_ldblockfile(ldblock_file):
        blocks = pd.read_csv(ldblock_file, sep='\t')
        blocks.columns = [k.strip() for k in blocks.columns]
        blocks['chr'] = blocks['chr'].str.strip('chr')
        blocks['chr'] = blocks['chr'].astype(int)
        return blocks

    def transform_sample_major(self, output: str) -> str:
        """
        Transform plink variant major to sample major.

        :param output:  output dir
        :return:  output path
        """
        assert self.sample_major is not True
        assert os.path.isdir(output)
        for p in self.plink_files:
            outpath = os.path.join(output, p+'_SampleMajor')
            command = [self.plink2_binary, '--bfile', p,
                       '--export', 'ind-major-bed', '--out', outpath]
            lg.debug('Used command:\n%s', command)
            subprocess.run(command)
        return output

    def train_dev_split(self, batch_size: int, frac: float = None, n_train: int = None):
        """
        Splits a fam file into training and dev set
        :param batch_size: int of the chosen batch size
        :param frac: fraction of training samples
        :param n_train:  absolute number of training samples
        :return: pandas data.frames with train and dev fam
        """
        assert (frac is not None) or (n_train is not None), 'Use either frac or n_train'
        assert (frac is None) or (n_train is None), 'Use either frac or n_train'
        columns = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO']
        fam = pd.read_table(self.plink_files[0]+'.fam', header=None, names=columns)
        n = fam.shape[0]
        lg.debug('Found %s samples in fam file', n)
        if frac is not None:
            mask = np.random.rand(n) < frac
            train = fam[mask]
            dev = fam[~mask]
        elif n_train is not None:
            sample_id = fam.index.values
            selected = np.random.choice(sample_id, n_train, replace=False)
            mask = np.array([x in selected for x in sample_id])
            train = fam[mask]
            dev = fam[~mask]
        train_n = train.shape[0]
        dev_n = dev.shape[0]
        lg.debug('split up into %s train and %s dev samples', train_n, dev_n)
        train_max = int(np.floor(train_n / batch_size) * batch_size)
        dev_max = int(np.floor(dev_n / batch_size) * batch_size)
        train = train[:train_max]
        dev = dev[:dev_max]
        del fam
        train.to_csv('.train.temp', index=None, header=None)
        dev.to_csv('.dev.temp', index=None, header=None)
        return train, dev

    def split_plink(self, output: str):
        assert os.path.isfile('.train.temp')
        assert os.path.isfile('.dev.temp')
        for p in self.plink_files:
            outpath = os.path.join(output, p+'_SampleMajor')
            train_command = [self.plink2_binary, '--bfile', p,
                             '--keep', '.train.temp',
                             '--export', 'ind-major-bed',
                             '--out', outpath+'_train']
            dev_command = [self.plink2_binary, '--bfile', p,
                           '--keep', '.dev.temp',
                           '--export', 'ind-major-bed',
                           '--out', outpath+'_dev']
            lg.debug('Used command:\n%s', train_command)
            lg.debug('Used command:\n%s', dev_command)
            subprocess.run(train_command)
            subprocess.run(dev_command)

    def split_plink_ldblock(self, output: str, sample_split: bool = True):
        """
        Splits plink file into LD blocks and wirites them in sample major format.

        :param output: output dir
        :param sample_split: bool if a sample split should be done in train and dev set
        :return: None
        """
        if sample_split:
            assert os.path.isfile('.train.temp')
            assert os.path.isfile('.dev.temp')
            for p in self.plink_files:
                bim = self._load_bim(p+'.bim')
                chromosomes = bim.chr.unique()
                for index, row in self.ldblocks.iterrows():
                    chr, start, end = row['chr'], row['start'], row['stop']
                    if chr not in chromosomes:
                        continue
                    lg.debug('Processing %s', row)
                    if not os.path.isdir(os.path.join(output, 'chr'+str(chr))):
                        os.mkdir(os.path.join(output, 'chr'+str(chr)))
                    file_name = [str(k) for k in ['SampleMajor', chr, start, end]]
                    file_name = '_'.join(file_name)
                    outpath = os.path.join(output, 'chr'+str(chr), file_name)
                    train_command = [self.plink2_binary, '--bfile', p,
                                     '--chr', str(chr),
                                     '--from-bp', str(start),
                                     '--to-bp', str(end),
                                     '--keep', '.train.temp',
                                     '--export', 'ind-major-bed',
                                     '--out', outpath+'_train']
                    dev_command = [self.plink2_binary, '--bfile', p,
                                   '--chr', str(chr),
                                   '--from-bp', str(start),
                                   '--to-bp', str(end),
                                   '--keep', '.dev.temp',
                                   '--export', 'ind-major-bed',
                                   '--out', outpath+'_dev']
                    lg.debug('Used command:\n%s', train_command)
                    lg.debug('Used command:\n%s', dev_command)
                    subprocess.run(train_command)
                    subprocess.run(dev_command)
        else:
            for p in self.plink_files:
                bim = self._load_bim(p+'.bim')
                chromosomes = bim.chr.unique()
                for index, row in self.ldblocks.iterrows():
                    lg.debug('Processing %s', row)
                    chr, start, end = row['chr'], row['start'], row['stop']
                    if chr not in chromosomes:
                        continue
                    file_name = [str(k) for k in [p, 'SampleMajor', chr, start, end]]
                    file_name = '_'.join(file_name)
                    outpath = os.path.join(output, file_name)
                    command = [self.plink2_binary, '--bfile', p,
                                     '--chr', chr, '--from-bp', start, '--to-bp', end,
                                     '--export', 'ind-major-bed',
                                     '--out', outpath+'_train']
                    lg.debug('Used command:\n%s', command)
                    subprocess.run(command)


