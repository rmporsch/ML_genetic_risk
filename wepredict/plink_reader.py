"""Read plink files."""
from pyplink import PyPlink
import pandas as pd
import os
import pickle
from typing import Any
import numpy as np


class Genetic_data_read(object):
    """Provides batch reading of plink Files."""

    def __init__(self, plink_file: str, batches):
        """Provide batch reading of plink Files."""
        super(Genetic_data_read, self).__init__()
        self.plink_file = plink_file
        self.plink_reader = PyPlink(plink_file)
        self.bim = self.plink_reader.get_bim()
        self.n = self.plink_reader.get_nb_samples()
        self.bim.columns = [k.strip() for k in self.bim.columns]
        self.chromosoms = self.bim.chrom.unique()
        if batches is not None:
            self._dirname = os.path.dirname(plink_file)
            self._pickel_path = plink_file+'.ld_blocks.pickel'
            if os.path.isfile(self._pickel_path):
                self.groups = pickle.load(open(self._pickel_path, 'rb'))
            else:
                self.groups = pd.read_csv(batches, sep='\t')
                self.groups.columns = [k.strip() for k in self.groups.columns]
                self.groups['chr'] = self.groups['chr'].str.strip('chr')
                self.groups['chr'] = self.groups['chr'].astype(int)
                self.groups = self._preprocessing_ldblock()
                pickle.dump(self.groups, open(self._pickel_path, 'wb'))
        else:
            self.groups = None

    def _preprocessing_ldblock(self) -> dict:
        if self.groups is None:
            return None
        else:
            out = {}
            for chr in self.chromosoms:
                subset_blocks = self.groups[self.groups.chr == chr]
                subset_bim = self.bim[self.bim.chrom == chr]
                out[chr] = []
                print(subset_bim.dtypes)
                print(subset_blocks.dtypes)
                for index, row in subset_blocks.iterrows():
                    start = row['start']
                    end = row['stop']
                    rsids = subset_bim[
                        (self.bim.pos >= start)
                        & (self.bim.pos <= end)
                         ].index.values
                    out[chr].append(rsids)
            return out

    def block_iter(self, chr: int = 22, if_save: bool = True) -> Any:
        """Block iteration."""
        assert chr in self.chromosoms
        current_block = 0
        block_ids = self.groups[chr][current_block]
        size_block = len(block_ids)
        genotypematrix = np.zeros((self.n, size_block), dtype=np.int8)
        pos_id = 0
        for snp, genotypes in self.plink_reader.iter_geno():
            if snp not in block_ids:
                continue
            else:
                genotypematrix[:, pos_id] = genotypes
                pos_id += 1
                if pos_id >= (size_block - 1):
                    if if_save:
                        savepath = str(chr)+'_LD_block_'+str(block_ids)+'.npy'
                        np.save(savepath)
                    yield genotypematrix
                    pos_id = 0
                    current_block += 1
                    block_ids = self.groups[chr][current_block]
                    size_block = len(block_ids)
                    genotypematrix = np.zeros((self.n, size_block),
                                              dtype=np.int8)
