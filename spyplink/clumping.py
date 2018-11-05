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
        lg.info('Outputting to %s', self.output_dir)

        self.plink2_binary = os.path.join(os.path.dirname(__file__),
                                          'bin/plink2')
        self.plink_binary = os.path.join(os.path.dirname(__file__),
                                         'bin/plink')
        lg.debug('location of plink2 file: %s', self.plink2_binary)
        lg.debug('location of plink file: %s', self.plink_binary)

    @staticmethod
    def _ispath(p: str):
        """
        Check file path

        :param p: path to file
        :return: None
        """
        if not os.path.exists(p):
            raise ValueError('Path %s does not exist' % p)

    def run_gwas(self, phenotype: str, mode: str = 'linear',
                 arguments: list = None):
        """
        Run GWAS wiht a given phenotype

        :param phenotype: phenotype name
        :param mode: mode ['linear', 'logistic']
        :param arguments: additional arguments
        :return: path to sumstat
        """

        assert mode in ['linear', 'logistic']

        output_files = list()
        for p in self.plink_files:
            nname = p.split('/')[-1]
            outpath = os.path.join(self.output_dir, nname)
            command = [self.plink2_binary, '--bfile', p,
                       '--pheno-name', phenotype,
                       '--pheno', self.pheno_path,
                       '--allow-no-sex',
                       '--'+mode, 'hide-covar',
                       '--out', outpath, *arguments]
            glm_path = '.'.join([outpath, phenotype, 'glm', mode])
            output_files.append([p, glm_path])
            lg.debug('Used command:\n%s', command)
            subprocess.run(command)
        return output_files

    @staticmethod
    def fix_gwas_output(input_path: str, output_path: str = None):
        """
        Fix summary stat file

        :param input_path: input file path
        :param output_path: output file path
        :return: None
        """
        if output_path is None:
            output_path = input_path
        temp_path = input_path+'.temp'
        command = ' '.join(['cat', input_path, '|', "tr",
                           '-s', "\' \'", "\'\\t\'", '>', temp_path])
        lg.debug('Pretty command: %s', command)
        subprocess.Popen(command, shell=True)
        command = ['mv', temp_path, output_path]
        lg.debug('Remove temp files: %s', command)
        subprocess.run(command)

    def _run_single_clumping(self, bfile: str, assoc_file: str,
                             p1: float, p2: float, r2: float):
        """
        Run a single clumping

        :param bfile: bfile stem
        :param assoc_file: summary stat
        :param p1:
        :param p2:
        :param r2:
        :return: file path to clumped sumstat
        """
        nname = assoc_file.split('/')[-1]
        output_path = os.path.join(self.output_dir, nname)
        command = [self.plink_binary,
                   '--bfile', bfile,
                   '--clump', assoc_file,
                   '--clump-snp-field', 'ID',
                   '--clump-p1', str(p1),
                   '--clump-p2', str(p2),
                   '--clump-r2', str(r2),
                   '--out', output_path]
        lg.debug('Clumping command %s', command)
        subprocess.run(command)
        self.fix_gwas_output(output_path+'.clumped')
        return output_path+'.clumped'

    def run_clumping(self, bfile_association: list, p1: float,
                     p2: float, r2: float):
        """
        Clump data

        :param bfile_association: list of lists with bfiles and sumstats
        :param p1:
        :param p2:
        :param r2:
        :return: outputfile paths as a list
        """
        output_files = []
        for p in bfile_association:
            bfile, assoc_file = p
            f = self._run_single_clumping(bfile, assoc_file,
                                          p1, p2, r2)
            output_files.append([bfile, f])

        return output_files
