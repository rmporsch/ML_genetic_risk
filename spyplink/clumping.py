import os
import subprocess
import logging
from spyplink.data_prep import DataPrep

lg = logging.getLogger(__name__)


class Clumping(DataPrep):

    def __init__(self, plink_file: str, pheno_path: str, output_dir: str,
                 ld_block_path: str = None):
        DataPrep.__init__(self, plink_file, ld_block_path)
        self._ispath(pheno_path)
        self._ispath(output_dir)
        self.output_dir = output_dir
        self.pheno_path = pheno_path

        self.plink2_binary = os.path.join(os.path.dirname(__file__),
                                          'bin/plink2')
        self.plink_binary = os.path.join(os.path.dirname(__file__),
                                         'bin/plink')
        lg.debug('location of plink2 file: %s', self.plink2_binary)
        lg.debug('location of plink file: %s', self.plink_binary)

    @staticmethod
    def _ispath(p: str):
        if not os.path.exists(p):
            raise ValueError('Path %s does not exist' % p)

    def run_gwas(self, phenotype: str, mode: str = 'linear',
                 arguments: list = None):

        assert mode in ['linear', 'logistic']

        output_files = list()
        for p in self.plink_files:
            nname = p.split('/')[-1]
            outpath = os.path.join(self.output_dir, nname, phenotype+'_gwas')
            command = [self.plink2_binary, '--bfile', p,
                       '--pheno', self.pheno_path,
                       '--allow-no-sex',
                       '--'+mode, 'hide-covar'
                       '--out', outpath, *arguments]
            output_files.append([p, outpath+'.assoc.'+mode])
            lg.debug('Used command:\n%s', command)
            subprocess.run(command)
        return output_files

    def _run_single_clumping(self, bfile: str, assoc_file: str,
                             output_path: str, p1: float,
                             p2: float, r2: float):
        nname = assoc_file.split('/')[-1]
        output_path = os.path.join(output_path, nname)
        command = [self.plink2_binary,
                   '--bfile', bfile,
                   '--clump', assoc_file,
                   '--clump-p1', p1,
                   '--clump-p2', p2,
                   '--clump-11', r2,
                   '--out', output_path]
        subprocess.run(command)
        return output_path+'.clumped'

    def run_clumping(self, bfile_association: list,
                     output_path: str, p1: float,
                     p2: float, r2: float):
        output_files = []
        for p in bfile_association:
            bfile, assoc_file = p
            f = self._run_single_clumping(bfile, assoc_file, output_path,
                                          p1, p2, r2)
            output_files.append(f)

        return output_files
