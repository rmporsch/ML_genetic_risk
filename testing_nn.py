import logging
from nnpredict.nnpredict import NNpredict


lg = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':

    pheno_file = 'data/sim_1000G_chr10.txt'
    batch_folder = 'data/sample_major_1kg/'
    ld_block_file = 'data/sim_1000G_chr10.ld_blocks.pickel'
    plink_file = 'data/sim_1000G_chr10'
    monster = NNpredict(ld_block_file, pheno_file, plink_file, batch_folder)
    sets = 'training_testingdivition.pickle'
    data_tt = 'testing_training_data.pickle'
    saved = True
    if saved:
        monster.set_training_testing(load_path=sets)
        monster.combine_testing_validation(load_path=data_tt)
    else:
        monster.set_training_testing(0.00, 0.20, save_path=sets)
        monster.combine_testing_validation(save_path=data_tt)
    monster.linear_model(penal=0.01, l_rate=0.01, name='weighted_sum_dropout2_lRg_BatchNorm-')
