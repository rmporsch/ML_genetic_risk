#!/usr/bin/env python
# -*- coding: utf-8 -*-
from spyplink.converting import Converting
import argparse
import logging
import pandas as pd

par = argparse.ArgumentParser(description='Convert plink files with summary statistics.')

par.add_argument('plinkpath', type=str,
                 help='Path to plink files (chr*)')

par.add_argument('output', type=str,
                 help='Output folder.')

par.add_argument('train', type=str,
                 help='path to train subjects')

par.add_argument('dev', type=str,
                 help='path to dev subjects')

par.add_argument('pheno_path', type=str,
                 help='Phenotype file')

par.add_argument('snps', type=str,
                 help='Sumstats path')

par.add_argument('-plink', '--plink',
                 type=str, dest='plink', default='spyplink/bin/plink',
                 help='Path to plink')

par.add_argument('-plink2', '--plink2',
                 type=str, dest='plink2', default='spyplink/bin/plink2',
                 help='Path to plink2')

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
    lg.debug('Plink file: %s', args.plinkpath)
    lg.debug('Output folder: %s', args.output)
    lg.debug('Pheno file: %s', args.pheno_path)
    lg.debug('SNP file: %s', args.snps)

    p = Converting(args.plinkpath, args.output, args.pheno_path)
    p.plink_binary = args.plink
    p.plink2_binary = args.plink2
    train = pd.read_table(args.train, header=None)
    dev = pd.read_table(args.dev, header=None)
    p.add_train_dev_split(train, dev)

    add_arguments = ['--extract', args.snps]
    major_files = p.split_plink(add_arguments)
    lg.info('Split output files %s', major_files)
