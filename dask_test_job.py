"""Compute LD-block wise prediction scores."""
from dask.distributed import Client, LocalCluster
from dask_jobqueue import PBSCluster
import pickle
import numpy as np
import pandas as pd
from wepredict.wepredict import wepredict
from glob import glob
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

def save_pickle(object, path):
    """Save obejct."""
    pickle.dump(object, open(path, 'wb'))


def load_pickle(path):
    """Save obejct."""
    return pickle.load(open(path, 'rb'))


if __name__ == '__main__':
    testing = False
    if testing:
        cluster = LocalCluster()
        cluster.scale(3)
        client = Client(cluster)
        print(client)
        folder = 'data/1kg_LD_blocks/'
        alphas = np.arange(0.2, 2, 0.4)
        monster = wepredict(folder+'/*', cluster, False)
        phenotype = monster.simulate()
        phenotype = phenotype.sum(axis=1).values
        index_valid = np.random.choice(range(len(phenotype)),
                                       100, replace=False)
        index_file = 'training_validation_index_testing.pickle'
        if os.path.isfile(index_file):
            index_valid = pickle.load(open(index_file, 'rb'))
        else:
            index_valid = np.random.choice(range(len(phenotype)), 100, replace=False)
            pickle.dump(index_valid, open(index_file, 'wb'))
        mask = np.ones(len(phenotype), dtype=bool)
        mask[index_valid] = False
        outfolder = ''
    else:
        cluster = PBSCluster(processes=2,
                threads=6, memory="120GB",
                queue='large',
                local_directory='$TMPDIR',
                resource_spec='select=1:ncpus=12:mem=120gb',
                walltime='12:00:00')
        print(cluster.job_script())
        cluster.scale(3)
        client = Client(cluster)
        print(client)

        folder = '/home2/groups/pcsham/users/rmporsch/sparse_matrix_ukb_chr10/'
        pheno_file = '/home/groups2/pcsham/users/rmporsch/simualted_ukb_chr10_phenotype/smaller_pheno.tab'
        alphas = np.arange(0.2, 2, 0.4)
        monster = wepredict(folder+'/10*', cluster, False)
        pheno = pd.read_csv(pheno_file, sep='\t')
        phenotype = pheno['V1'].values
        files = glob(folder+'/10*')
        index_file = 'training_valid_index.pickle'
        if os.path.isfile(index_file):
            index_valid = pickle.load(open(index_file, 'rb'))
        else:
            index_valid = np.random.choice(range(len(phenotype)), len(phenotype)*0.1, replace=False)
            pickle.dump(index_valid, open(index_file, 'wb'))
        mask = np.ones(len(phenotype), dtype=bool)
        mask[index_valid] = False
        outfolder = '/home2/groups/pcsham/users/rmporsch'


    models = ['l1', 'l2', 'l0']
    for norm in models:
        model_save = os.path.join(outfolder, 'models_'+norm+'.pickle')
        eval_save = os.path.join(outfolder, 'eval_'+norm+'.pickle')
        monster.generate_DAG(phenotype, index_valid, alphas, norm)
        out = monster.compute()
        save_pickle(out, model_save)
        oo = monster.evaluat_blocks(out, phenotype[~mask])
        save_pickle(oo, eval_save)
    # cluster.close()
