import logging
from glob2 import glob
import os
import filecmp
from typing import List, Tuple, Dict
BoolVector = List[bool]
import pandas as pd
import numpy as np

lg = logging.getLogger(__name__)


class DataPrep(object):

    def __init__(self, plink_path: str, ld_block_path: str = None):
        """
        DataPrep class to collect and prepare input.

        :param plink_path: path to plink files
        :param ld_block_path: path to block files
        """
        self._plink_path = plink_path
        self._ld_block_path = ld_block_path
        self.plink_files = self._expand_path(plink_path)
        lg.info('Using the following plink files:\n%s',
                self.plink_files)
        self.sample_major = self._check_files(self.plink_files)
        if ld_block_path is not None:
            self.ldblocks = self._load_ldblockfile(ld_block_path)
        else:
            self.ldblocks = None

    @staticmethod
    def _expand_path(plink_path: str) -> List:
        """
        Search and list plink file paths

        :param plink_path: path to plink file with wildcard
        :return: list of plinks path
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

    def _check_files(self, plink_path: List) -> bool:
        """
        Check plink file

        :param plink_path: plink file path
        :return: status of the file
        """
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
    def _check_plink_type(plink_path: str) -> bool:
        """
        Check plink type for sample major format

        :param plink_path: plink file path
        :return: bool, true for sample major
        """
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
    def _load_ldblockfile(ldblock_file) -> Dict:
        """
        Load block file from bed file

        :param ldblock_file: bed file path
        :return: dictionary of blocks
        """
        blocks = pd.read_csv(ldblock_file, sep='\t')
        blocks.columns = [k.strip() for k in blocks.columns]
        blocks['chr'] = blocks['chr'].str.strip('chr')
        blocks['chr'] = blocks['chr'].astype(int)
        return blocks

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
        train.to_csv('.train.temp', index=None, header=None, sep='\t')
        dev.to_csv('.dev.temp', index=None, header=None, sep='\t')
        return train, dev

    @staticmethod
    def add_train_dev_split(train, dev):
        """
        Short function to write train and dev to disk in required format.

        :param train: pandas DataFrame of train
        :param dev: pandas DataFrame of dev
        :return: None
        """
        train.to_csv('.train.temp', index=None, header=None, sep='\t')
        dev.to_csv('.dev.temp', index=None, header=None, sep='\t')
