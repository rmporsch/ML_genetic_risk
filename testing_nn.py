import logging
from nnpredict.nnpredict import NNpredict


lg = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':

    pheno_file = 'data/sim_1000G_chr10.txt'
    train_plink = 'data/split_test/SampleMajor_train'
    dev_plink = 'data/split_test/SampleMajor_dev'
    ld_block_file = 'data/sim_1000G_chr10.ld_blocks.pickel'
    monster = NNpredict(train_plink, dev_plink, pheno_file, ld_block_file)

    monster.linear_model(epochs=10, batch_size=100,
                         penal=0.01, l_rate=0.01, pheno_name='V1',
                         tb_name='sample_majorMode')
