"""Convert plink file into numpy format by batches."""

import logging
import argparse
import wepredict.plink_reader as pr

lg = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='Convert plink files to numpy')
parser.add_argument('--plink', required=True,
                    type=str, help='path to plink stem')
parser.add_argument('--ldblock',required=True,
                    type=str, help='path to LD block file')
parser.add_argument('--pheno', required=True,
                    type=str, help='path to pheno file')
parser.add_argument('--output', required=True,
                    type=str, help='path to output folder')
parser.add_argument('--batchsize', required=True,
                    type=int, help='size of mini-batches')
parser.add_argument('--debug', required=False, default=False,
                    help='debugging')

args = parser.parse_args()

if args.debug:
    lg.setLevel(logging.DEBUG)


if __name__ == '__main__':
    plink_file = args.plink
    LD_file = args.ldblock
    pheno_file = args.pheno
    output_folder = args.output
    batch_size = args.batchsize

    pp = pr.Genetic_data_read(plink_file, LD_file, pheno_file)
    pp.rewrite(batch_size, output_folder)
