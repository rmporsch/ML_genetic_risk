from dask.distributed import Client, LocalCluster
from dask import delayed
import dask
from wepredict.pytorch_regression import pytorch_linear
from wepredict.plink_reader import Genetic_data_read, get_genotypes
from wepredict.helpers import *


if __name__ == '__main__':
    plink_file = 'data/chr10'
    pheno_file = 'data/sim_1000G_chr10.txt'
    ld_block_file = 'data/Berisa.EUR.hg19.bed'
    data = Genetic_data_read(plink_file, ld_block_file, pheno_file)
    train_index, valid_index, test_index = generate_valid_test_data(data.n,
                                                                    0.1, 0.1)

    cluster = LocalCluster()
    w = cluster.start_worker(ncors=2)
    client = Client(cluster)
    print(client)
    out = list()
    for ld_block in data.groups[10]:
        mat = delayed(get_genotypes)(22, ld_block, plink_file, data.sub_in)
        sample = delayed(get_samples)(mat, data.pheno['V1'].values,
                                      train_index, valid_index)
        model = delayed(pytorch_linear)(sample['training_x'],
                                        sample['training_y'],
                                        sample['valid_x'],
                                        sample['valid_y'],
                                        mini_batch_size=100, type='c')
        results = delayed(model.run)('l1', lamb=0.01, epochs=100)
        out.append(results)

    res = dask.compute(out)
    cluster.close()
    print('done')