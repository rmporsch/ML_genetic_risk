"""
Class to prep data
"""
import os
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple
import subprocess
from spyplink.clumping import Clumping

lg = logging.getLogger(__name__)
BoolVector = List[bool]
SetTuple = Tuple[str, str]


class Converting(Clumping):

    def __init__(self, plink_path: str, output_dir: str,
                 pheno_file: str = None, ld_block_path: str = None):
        """
        Preprocessing of plink files for ML.

        :param plink_path: path to plink stem or stems (use *)
        :param output_dir: path to output
        :param pheno_file: path to phenotypes
        :param ld_block_path: path to LD blocks in bed format
        """
        Clumping.__init__(self, plink_path, pheno_file,
                          output_dir, ld_block_path)

    @staticmethod
    def _load_bim(bimfile):
        """
        Load bim file and add headers to it

        :param bimfile:
        :return: bim file as DataFrame
        """
        headernames = ['chr', 'rsid', 'm', 'bp', 'a1', 'a2']
        bim = pd.read_table(bimfile, header=None, names=headernames)
        lg.debug('dtype of chr in bim file is %s', bim.chr.dtype)
        if (bim.chr.dtype == np.float) or (bim.chr.dtype == np.int):
            return bim
        else:
            bim['chr'] = bim['chr'].str.strip('chr')
            bim['chr'] = bim['chr'].astype(int)
            return bim

    def convert_sample_major(self, bfile, name: str = None, args: List = None) -> str:
        """
        Transform plink variant major to sample major.

        :param bfile: plink file
        :param args: additional arguments as a list
        :return:  output path
        """
        nname = bfile.split('/')[-1]
        fname = nname+'_SampleMajor'+name
        out_path = os.path.join(self.output_dir, fname)
        command = [self.plink2_binary,
                   '--bfile', bfile,
                   '--export', 'ind-major-bed',
                   '--out', out_path, *args]
        lg.debug('Used command:\n%s', command)
        subprocess.run(command)
        return out_path

    def split_plink(self, args: List = None) -> List:
        """
        Split plink file into train and dev set.

        :param args: additional arguments as a list
        :return: List of output files
        """
        assert os.path.isfile('.train.temp')
        assert os.path.isfile('.dev.temp')
        train = ['--keep', '.train.temp', *args]
        dev = ['--keep', '.dev.temp', *args]
        out_list = list()
        for p in self.plink_files:
            plink_train = self.convert_sample_major(p, 'train', train)
            plink_dev = self.convert_sample_major(p, 'dev', dev)
            out_list.append([plink_train, plink_dev])
        return out_list

