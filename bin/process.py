from spyplink.converting import Converting
import argparse
import logging
import pandas as pd

par = argparse.ArgumentParser(description='Convert plink files.')

par.add_argument('plinkpath', type=str,
                 help='Path to plink files (chr*)')

par.add_argument('output', type=str,
                 help='Output folder.')

par.add_argument('train', type=int,
                 help='batch size')

par.add_argument('dev', type=float,
                 help='Total number of training samples or fraction of n')

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
    train = pd.read_table(args.train)
    dev = pd.read_table(args.dev)
    p.add_train_dev_split(train, dev)
    sumstat = p.run_gwas(args.pheno)
    clumped_files = p.run_clumping(sumstat, args.output,
                                   args.p1, args.p2, args.r2)
    lg.info('Clumped files: %s', clumped_files)
    for p, ss in clumped_files:
        snp_list = pd.read_table(ss)
        lg.debug('%s', snp_list.head())
        p.convert_sample_major()


