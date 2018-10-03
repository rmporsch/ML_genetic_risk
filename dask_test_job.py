"""Compute LD-block wise prediction scores."""
from dask.distributed import Client, LocalCluster
from dask_jobqueue import PBSCluster
import pickle
import numpy as np
import pandas as pd
from weepredict.helpter_function import save_pickle
from wepredict.wepredict import wepredict
from pyplink_major.plink_reader import Genetic_data_read
from glob import glob
import os
import sys
import socket
sys.path.insert(0, os.path.abspath('.'))

if __name__ == '__main__':
    hostname = socket.gethostname()
    if hostname == 'rmporsch':
        print('Running locally')
        cluster = LocalCluster()
        cluster.scale(3)
        client = Client(cluster)
        print(client)
        folder = 'data/1kg_LD_blocks/'
        plink_file = 'data/1kg_phase1_chr22'
        ld_blocks_file = 'data/Berisa.EUR.hg19.bed'
        plink_reader = Genetic_data_read(plink_file, ld_blocks_file)
        alphas = np.arange(0.2, 2, 0.4)
        monster = wepredict(folder+'/*', True)
        phenotype = monster.simulate
        n = len(phenotype)
        phenotype = phenotype.sum(axis=1).values
        index_file = 'training_validation_index_testing.pickle'
        if os.path.isfile(index_file):
            index = pickle.load(open(index_file, 'rb'))
        else:
            index = monster.generate_valid_test_data(n, 0.2, 0.1)
            pickle.dump(index, open(index_file, 'wb'))
        outfolder = ''
    else:
        cluster = PBSCluster(processes=4,
                cores=5, memory="120GB",
                queue='large',
                local_directory='$TMPDIR',
                resource_spec='select=1:ncpus=12:mem=120gb',
                walltime='84:00:00')
        print(cluster.job_script())
        cluster.scale(3)
        client = Client(cluster)
        print(client)

        folder = '/home2/groups/pcsham/users/rmporsch/sparse_matrix_ukb_chr10/'
        pheno_file = '/home/groups2/pcsham/users/rmporsch/simualted_ukb_chr10_phenotype/smaller_pheno.tab'
        lambfile = '/home/tshmak/WORK/Projects/bigdata/ForRobert/lasso_benchmark_lambda.txt'
        with open(lambfile, 'r') as f:
            alphas = []
            next(f)
            for line in f:
                alphas.append(float(line))
        n_alphas = 10
        alphas = np.random.choice(alphas, n_alphas, replace=False)
        monster = wepredict(folder+'/10*', cluster, False)
        pheno = pd.read_csv(pheno_file, sep='\t')
        phenotype = pheno['V1'].values
        n = len(phenotype)
        files = glob(folder+'/10*')
        index_file = 'training_valid_index.pickle'
        if os.path.isfile(index_file):
            index = pickle.load(open(index_file, 'rb'))
        else:
            index = monster.generate_valid_test_data(n, 0.2, 0.1)
            pickle.dump(index, open(index_file, 'wb'))
        outfolder = '/home2/groups/pcsham/users/rmporsch'

    models = ['l1', 'l2', 'l0']
    models = ['l1', 'l0']
    param = {'l1': {'mini_batch': 10000, 'l_rate': 0.0001, 'epochs': 201},
             'l2': {'mini_batch': 10000, 'l_rate': 0.0001, 'epochs': 201},
             'l0': {'mini_batch': 10000, 'l_rate': 0.001, 'epochs': 201}}
    index_training, index_valid, index_test = index
    for norm in models:
        specific_param = param[norm]
        model_save = os.path.join(outfolder, 'models_'+norm+'.pickle')
        eval_save = os.path.join(outfolder, 'eval_'+norm+'.pickle')
        monster.generate_DAG(phenotype, index_training,
                             index_valid, alphas,
                             norm, **specific_param)
        out = monster.compute()
        save_pickle(out, model_save)
        oo = monster.evaluat_blocks(out, phenotype[index_valid])
        save_pickle(oo, eval_save)
