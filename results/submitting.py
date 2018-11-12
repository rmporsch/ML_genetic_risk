import logging
import subprocess
import time
import numpy as np
import argparse
import os

par = argparse.ArgumentParser(description='Submit models.')
par.add_argument('model', choices=['clumped', 'ldfull', 'lassosum'],
        help='Which model shall it be?')
par.add_argument('data', choices=['ukb', 'onekg'],
        help='Which data shall it be?')
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
    sumstats = {'ukb': '/home/rmporsch/projects/ML_genetic_risk/data/sample_major/ukb/clumped/maf_0.01_10.V1.glm.linear.clumped.merged',
            'onekg': ''}
    ref = '/home/rmporsch/projects/ML_genetic_risk/data/sim_1000G_chr10' 
    dev_variant_major = \
            {'ukb': '/home2/groups/pcsham/users/rmporsch/ml_data/sample_major/ukb/clumped/lassosum/maf_0.01_10_VariantMajordev',
            'onekg': '/home/rmporsch/projects/ML_genetic_risk/results/lassosum/1kg_variant_major_dev'}
    pheno_paths = {'ukb': '/home/rmporsch/projects/ML_genetic_risk/data/simulated_chr10.txt',
            'onekg': '/home/rmporsch/projects/ML_genetic_risk/data/sim_1000G_chr10.txt'}
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

