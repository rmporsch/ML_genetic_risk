import logging
import subprocess
import time
import numpy as np
import argparse
import os

par = argparse.ArgumentParser(description='Submit models.')
par.add_argument('train', type=str, help='path to train data (plink/sumstat)')
par.add_argument('dev', type=str, help='path to dev data')
par.add_argument('pheno', type=str, help='path to pheno data')
par.add_argument('model', choices=['clumped', 'ldfull', 'lassosum'],
        help='Which model shall it be?')
par.add_argument('-nl', default=1, dest='nl',
        type=int, help='number of lambdas')
par.add_argument('-p', default='V1', dest='p',
        type=str, help='phenotype')

par.add_argument("-v", "--verbose", action="store_const", dest="log_level",
                const=logging.INFO, default=logging.WARNING)
par.add_argument("-d", "--debug", action="store_const", dest="log_level",
                 const=logging.DEBUG)

args = par.parse_args()
logging.basicConfig(level=args.log_level)
lg = logging.getLogger(__name__)

def job_string(job_name: str, walltime: str, processors: str, command: str, queue: str):
    job_string = """#!/bin/bash
        #PBS -N %s
        #PBS -l walltime=%s
        #PBS -l %s
        #PBS -l mem=8gb
        #PBS -q %s
        cd /home/rmporsch/projects/ML_genetic_risk
        %s""" % (job_name, walltime, processors, queue, command)
    return job_string

if __name__ == '__main__':

    phenotypes = ['V'+str(k) for k in range(1, 5)]
    phenotypes = ['V1']
    lg.info('Using the following phenotypes %s', phenotypes)
    walltime = '6:00:00'
    processors = 'nodes=1:ppn=2'
    queue = 'small'
    num_lambda = args.nl
    model = {'clumped': '/home/rmporsch/projects/ML_genetic_risk/results/clumped.py',
            'ldfull': '/home/rmporsch/projects/ML_genetic_risk/results/ldfull.py',
            'lassosum': '/home/rmporsch/projects/ML_genetic_risk/results/lassosum/lassosum.r'}
    sample_sizes = {'ukb': 285000, 'onekg': 2200}
    ref = '/home/rmporsch/projects/ML_genetic_risk/data/sim_1000G_chr10' 
    lambdas = np.exp(np.linspace(np.log(0.01), np.log(0.1), num=num_lambda))
    lg.debug('Using %s lambdas', lambdas)

    subfile = open('.pbs.submit', 'w')
    if args.model in ['clumped', 'ldfull']:
        assert len(lambdas) == 1
        more_args = [args.p, args.data, str(lambdas[0])]
        command = ' '.join(['python ', model[args.model], *more_args])
    if args.model == 'lassosum':
        assert args.data != 'onekg'
        ref_genome = ref
        dev_plink = dev_variant_major[args.data]
        sumstat = sumstats[args.data]
        ldfile = '/home/rmporsch/projects/ML_genetic_risk/data/Berisa.EUR.hg19.bed'
        n = sample_sizes[args.data]
        command = ' '.join([
            'Rscript', model[args.model], ref_genome, dev_plink,
            sumstat, ldfile, str(n), pheno_paths[args.data], args.p])

    subfile = open('.pbs.submit', 'w')
    job_name = '_'.join([args.data, args.p, args.model])
    string = job_string(job_name, walltime, processors, command, queue)
    lg.debug(string)
    subfile.write(string)
    subfile.close()
    subprocess.call(['qsub', '.pbs.submit'])
    # subprocess.call(['bash', '.pbs.submit'])
    time.sleep(0.1)

