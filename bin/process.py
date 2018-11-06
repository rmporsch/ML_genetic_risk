from spyplink.converting import Converting
import argparse
import logging
import pandas as pd

par = argparse.ArgumentParser(description='Convert plink files.')

par.add_argument('plinkpath', type=str,
                 help='Path to plink files (chr*)')

par.add_argument('output', type=str,
                 help='Output folder.')

par.add_argument('train', type=str,
                 help='path to train subjects')

par.add_argument('dev', type=str,
                 help='path to dev subjects')

par.add_argument('pheno', type=str,
                 help='Phenotype to analyse')

par.add_argument('-p1', default=0.01, type=float, dest='p1',
                 help='clumping threshold for p1')

par.add_argument('-p2', default=0.01, type=float, dest='p2',
                 help='clumping threshold for p1')

par.add_argument('-r2', default=0.01, type=float, dest='r2',
                 help='clumping threshold for p1')

par.add_argument('-ld', '--ldpath',
                 type=str, dest='ldblocks',
                 help='Path to ldblocks')

par.add_argument("-v", "--verbose", action="store_const",
                 dest="log_level", const=logging.INFO,
                 default=logging.WARNING)

par.add_argument("-d", "--debug",
                 action="store_const", dest="log_level",
                 const=logging.DEBUG)

args = par.parse_args()
logging.basicConfig(level=args.log_level)
lg = logging.getLogger(__name__)


if __name__ == '__main__':
    p = Converting(args.plinkpath, args.output, args.pheno)
    assert len(p.plink_files) == 1
    train = pd.read_table(args.train)
    dev = pd.read_table(args.dev)
    p.add_train_dev_split(train, dev)
    sumstat = p.run_gwas(args.pheno)
    clumped_files = p.run_clumping(sumstat, args.p1, args.p2, args.r2)
    lg.info('Clumped files: %s', clumped_files)
    clumped_gwas = pd.read_table(clumped_files[0][1])
    snps = clumped_gwas.SNP
    snps.to_csv('.snps.list', index=False, header=None)
    add_arguments = ['--extract', '.snps.list']
    major_files = p.split_plink(add_arguments)
    lg.info('Split output files %s', major_files)
