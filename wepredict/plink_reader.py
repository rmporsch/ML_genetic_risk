"""Read plink files."""
from pyplink import PyPlink
import pandas as pd
import os
import pickle
from typing import Any
import numpy as np
from scipy import sparse
import logging

lg = logging.getLogger(__name__)

def get_genotypes(self, chr, rsid, plink_path, sub_in):
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


class Genetic_data_read(object):
    """Provides batch reading of plink Files."""

    def __init__(self, plink_file: str, batches, pheno: str = None):
        """Provide batch reading of plink Files."""
        super(Genetic_data_read, self).__init__()
        self.plink_file = plink_file
        self.plink_reader = PyPlink(plink_file)
        self.bim = self.plink_reader.get_bim()
        self.n = self.plink_reader.get_nb_samples()
        self.fam = self.plink_reader.get_fam()
        self.bim.columns = [k.strip() for k in self.bim.columns]
        self.chromosoms = self.bim.chrom.unique()
        self.pheno_file = pheno
        lg.debug('Chromosom is in the following format %s', self.chromosoms[0])
        if batches is not None:
            self._dirname = os.path.dirname(plink_file)
            self._pickel_path = plink_file+'.ld_blocks.pickel'
            if os.path.isfile(self._pickel_path):
                with open(self._pickel_path, 'rb') as f:
                    self.groups = pickle.load(f)
            else:
                self.groups = pd.read_csv(batches, sep='\t')
                self.groups.columns = [k.strip() for k in self.groups.columns]
                self.groups['chr'] = self.groups['chr'].str.strip('chr')
                self.groups['chr'] = self.groups['chr'].astype(int)
                self.groups = self._preprocessing_ldblock()
                pickle.dump(self.groups, open(self._pickel_path, 'wb'))
        else:
            self.groups = None
        lg.debug('Group is a %s with the following keys %s',
                 type(self.groups), self.groups.keys())
        if pheno is not None:
            self.pheno = self._process_pheno(pheno)
        else:
            lg.warning('No phenotype given, using fam file instead.')
            self.pheno = self.fam
            self.sub_in = np.ones(self.n, bool)
            self.subject_ids = self.fam['iid'].values
        self.plink_reader.close()

    def _process_pheno(self, pheno_file):
        pheno = pd.read_table(pheno_file)
        import_n, import_p = pheno.shape
        lg.debug('loaded phenotype with the shape %s', pheno.shape)
        header = pheno.columns.values
        lg.debug('Phenotype has the following headers %s', header)
        assert len(header) > 1
        assert np.all([k in header for k in ['IID', 'FID']]) or \
            np.all([k in header for k in ['iid', 'fid']])
        if 'IID' in header:
            pheno.rename(columns={'IID': 'iid', 'FID': 'fid'},
                         inplace=True)
        if np.all([k in header for k in ['PAT', 'MAT', 'SEX']]):
            pheno.drop(['PAT', 'MAT', 'SEX'], axis=1, inplace=True)
        pheno[['iid', 'fid']] = pheno[['iid', 'fid']].astype(str)
        new_fam = self.fam.merge(pheno, 'inner', on=['iid', 'fid'])
        merged_n, merged_p = new_fam.shape
        lg.debug('Format of the new fam file %s', new_fam.shape)
        if merged_n < import_n:
            lg.warning('Out of %s subjects, %s were in fam file',
                    import_n, merged_n)
            if merged_n == 0:
                raise ValueError('No subject present in file')
        pheno_columns = [k in ['fid', 'iid'] for k in new_fam.columns.values]
        pheno_columns = new_fam.columns.values[~np.array(pheno_columns)]
        lg.info('Extracted and integrated the following phenotypes %s',
                pheno_columns)
        self.pheno_names = pheno_columns
        self.subject_ids = new_fam['iid'].values
        self.sub_in = [k in new_fam['iid'].values for k in self.fam['iid']]
        self.n = sum(self.sub_in)
        return new_fam


    def _preprocessing_ldblock(self) -> dict:
        if self.groups is None:
            return None
        else:
            out = {}
            for chr in self.chromosoms:
                subset_blocks = self.groups[self.groups.chr == chr]
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
