from wepredict.pytorch_regression import pytorch_linear
from wepredict.plink_reader import Genetic_data_read, get_genotypes
from wepredict.helpers import *
import sys


if __name__ == '__main__':
    plink_file = 'data/chr10'
    pheno_file = 'data/sim_1000G_chr10.txt'
    ld_block_file = 'data/Berisa.EUR.hg19.bed'
    data = Genetic_data_read(plink_file, ld_block_file, pheno_file)
    train_index, valid_index, test_index = generate_valid_test_data(data.n,
                                                                    0.1, 0.1)

    ld_block = data.groups[10][0]
    mat = get_genotypes(22, ld_block, plink_file, data.sub_in)
    sample = get_samples(mat, data.pheno['V1'].values,
                                  train_index, valid_index)
    model = pytorch_linear(sample['training_x'],
                                    sample['training_y'],
                                    sample['valid_x'],
                                    sample['valid_y'],
                                    mini_batch_size=100, type='c')
    results = model.run('l1', lamb=0.01, epochs=100)
