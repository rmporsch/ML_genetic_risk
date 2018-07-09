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
    folder = 'data/1kg_LD_blocks/'
    alphas = np.arange(0.2, 2, 0.4)

    cluster = LocalCluster()
    cluster.scale(3)
    client = Client(cluster)
    print(client)

    monster = wepredict(folder+'/*', cluster, True)
    phenoype = monster.simulate()
    phenoype = phenoype.sum(axis=1).values
    index_valid = np.random.choice(range(len(phenoype)), 100, replace=False)
    mask = np.ones(len(phenoype), dtype=bool)
    mask[index_valid] = False

    monster.generate_DAG(phenoype, index_valid, alphas, 'l1')
    out = monster.compute()
    save_pickle(out, 'testing.pickle')
    # out = load_pickle('testing.pickle')
    # import pdb; pdb.set_trace()
    oo = monster.evaluat_blocks(out, phenoype[~mask])
    print(oo)
    cluster.close()
