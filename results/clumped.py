import os
from nnpredict.nnpredict import NNpredict
from nnpredict.models import LinearModel,NNModel
import logging
import sys
import numpy as np

lg = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def kg_run(pheno: str ='V1', penal: float = 0.01):

    pheno_file = 'data/sim_1000G_chr10.txt'
    train_plink = 'data/sample_major/1kg/clumped/sim_1000G_chr10_SampleMajor_train'
    dev_plink = 'data/sample_major/1kg/clumped/sim_1000G_chr10_SampleMajor_dev'

    ld_blocks = 'data/sim_1000G_chr10.ld_blocks.pickel'
    tb_path = 'results/tensorboard/clumped'
    outputdir = 'results/clumped'

    model = NNpredict(train_plink, dev_plink, pheno_file, tb_path)

    model.run_model(epochs=350,
            batch_size=100,
            l_rate=0.0001,
            penal=penal,
            pheno_name=pheno,
            tb_name=pheno+'_1kg',
            in_model=NNModel,
            layers=[85, 60, 20, 10],
            export_dir=outputdir)

def ukb_run(pheno: str = 'V1', penal: float = 0.01):

    pheno_file = 'data/simulated_chr10.txt'
    train_plink = 'data/sample_major/ukb/clumped/maf_0.01_10_SampleMajortrain'
    dev_plink = 'data/sample_major/ukb/clumped/maf_0.01_10_SampleMajordev'

    ld_blocks = 'data/maf_0.01_10.ld_blocks.pickel'
    tb_path = 'results/tensorboard/clumped'
    outputdir = 'results/clumped'

    model = NNpredict(train_plink, dev_plink, pheno_file, tb_path)

    model.run_model(epochs=350,
            batch_size=1000,
            l_rate=0.0001,
            penal=penal,
            pheno_name=pheno,
            tb_name=pheno+'_ukb',
            in_model=NNModel,
            layers=[85, 60, 20, 10],
            export_dir=outputdir)

if __name__ == '__main__':
    os.chdir('/home/rmporsch/projects/ML_genetic_risk')
    args = (sys.argv)
    data = {'ukb': ukb_run, 'onekg': kg_run}
    if len(args) > 1:
        p = args[1]
        run = data[args[2]]
        l = float(args[3])
        run(p, l)
    else:
        raise ValueError('need at least 3 parameters')

