import os
from nnpredict.nnpredict import NNpredict
from nnpredict.models import LinearModel
import logging
import sys

lg = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def kg_run(pheno: str ='V1', penal: float = 0.01):

    pheno_file = 'data/sim_1000G_chr10.txt'
    train_plink = 'data/sample_major/1kg/sim_1000G_chr10_SampleMajor_train'
    dev_plink = 'data/sample_major/1kg/sim_1000G_chr10_SampleMajor_dev'

    ld_blocks = 'data/sim_1000G_chr10.ld_blocks.pickel'
    tb_path = 'results/tensorboard/LDfull'
    outputdir = 'results/ldfull' 
    model = NNpredict(train_plink, dev_plink, pheno_file, tb_path)

    model.run_model(epochs=350,
            batch_size=100,
            l_rate=0.0001,
            penal=penal,
            pheno_name=pheno,
            tb_name=pheno+'_1kg',
            in_model=LinearModel,
            ld_blocks=ld_blocks,
            export_dir=outputdir)

def ukb_run(pheno: str = 'V1', penal: float = 0.01):

    pheno_file = 'data/simulated_chr10.txt'
    train_plink = 'data/sample_major/ukb/maf_0.01_10_SampleMajor_train'
    dev_plink = 'data/sample_major/ukb/maf_0.01_10_SampleMajor_dev'

    ld_blocks = 'data/maf_0.01_10.ld_blocks.pickel'
    tb_path = 'results/tensorboard/LDfull'
    outputdir = 'results/ldfull'

    model = NNpredict(train_plink, dev_plink, pheno_file, tb_path)

    model.run_model(epochs=350,
            batch_size=1000,
            l_rate=0.0001,
            penal=penal,
            pheno_name=pheno,
            tb_name=pheno+'_ukb',
            in_model=LinearModel,
            ld_blocks=ld_blocks,
            export_dir=outputdir)

if __name__ == '__main__':
    os.chdir('/home/rmporsch/projects/ML_genetic_risk')
    args = (sys.argv)
    if len(args) > 1:
        p = args[1]
        if len(args) == 3:
            kg_run(p, float(args[2]))
        else:
            kg_run(p)
    else:
        kg_run()
