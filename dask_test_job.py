"""Compute LD-block wise prediction scores."""
from dask.distributed import Client, LocalCluster
import pickle
import numpy as np
from wepredict.wepredict import wepredict

def save_pickle(object, path):
    """Save obejct."""
    pickle.dump(object, open(path, 'wb'))


def load_pickle(path):
    """Save obejct."""
    return pickle.load(open(path, 'rb'))


if __name__ == '__main__':
    testing = True
    if testing:
        cluster = LocalCluster()
        cluster.scale(3)
        client = Client(cluster)
        print(client)
        folder = 'data/1kg_LD_blocks/'
        alphas = np.arange(0.2, 2, 0.4)
        monster = wepredict(folder+'/*', cluster, True)
        phenoype = monster.simulate()
        phenoype = phenoype.sum(axis=1).values
        index_valid = np.random.choice(range(len(phenoype)),
                                       100, replace=False)
        mask = np.ones(len(phenoype), dtype=bool)
        mask[index_valid] = False
        save_pickle(index_valid, 'training_validation_index.pickle')
    else:
        folder = ''
        alphas = np.arange(0.2, 2, 0.4)

    models = ['l1', 'l2', 'l0']
    for norm in models:
        monster.generate_DAG(phenoype, index_valid, alphas, norm)
        out = monster.compute()
        save_pickle(out, 'models_'+norm+'.pickle')
        oo = monster.evaluat_blocks(out, phenoype[~mask])
        save_pickle(oo, 'eval_'+norm+'.pickle')
    cluster.close()
