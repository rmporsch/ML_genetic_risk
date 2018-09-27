#!/bin/python3
"""
Script to convert plink files into sample major format and more
"""
from pyplink_major.data_processing import PreProcess
import argparse
import logging

par = argparse.ArgumentParser(description='Convert plink files.')

par.add_argument('plinkpath', type=str,
                 help='Path to plink files (chr*)')

par.add_argument('output', type=str,
                 help='Output folder.')

par.add_argument('-ld', '--ldpath',
                 type=str, dest='ldblocks',
                 help='Path to ldblocks')

par.add_argument('batchsize', type=int,
                 help='batch size')

par.add_argument('trainsize', type=float,
                 help='Total number of training samples or fraction of n')

par.add_argument("-v", "--verbose", action="store_const",
                 dest="log_level", const=logging.INFO,
                 default=logging.WARNING)

par.add_argument("-d", "--debug",
                 action="store_const", dest="log_level",
                 const=logging.DEBUG)
args = par.parse_args()
logging.basicConfig(level=args.log_level)


if __name__ == '__main__':
    model = PreProcess(args.plinkpath, args.ldblocks)
    if args.trainsize.is_integer():
        train, dev = model.train_dev_split(args.batchsize,
                                           n_train=args.trainsize)
    else:
        train, dev = model.train_dev_split(args.batchsize,
                                           frac=args.trainsize)
    model.split_plink_ldblock(output=args.output)