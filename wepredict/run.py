"""Read plink files by predefined chuncks."""
import pandas as pd
import numpy as np
import os
from pytorch_regression import pytorch_linear
from tensorflow_penal_regression import tensorflow_models
from sklearn_penal_regression import sklearn_models
from sklearn.preprocessing import scale
from typing import Any
from plink_reader import Genetic_data_read
from data_download import genetic_testdata


class DataProcessing(object):
    """Processe phenotypes."""

    def __init__(self, pheno: str):
        """Process phenotypes."""
        super(DataProcessing, self).__init__()
        assert isinstance(pheno, str)
        assert os.path.isfile(pheno)
        self.pheno_path = pheno

    def get_pheno(self, x: int) -> Any:
        """Get a specific phenotype."""
        assert x > 0
        assert isinstance(x, int)
        smaller_pheno_file = os.path.join("/tmp",
                                          'smaller_pheno.tab')
        os.system('cut -f1,2,3,4,5,{} {} > {}'.format(str(x+6),
                                                      self.pheno_path,
                                                      smaller_pheno_file))

        pheno = pd.read_csv(smaller_pheno_file, sep='\t')
        pheno = pheno.rename(columns={pheno.columns.values[-1]: 'pheno'})
        return pheno


if __name__ == '__main__':
    # Downloads
    download_path = 'data/'
    sim_path = 'data/phenotypes/simulated_chr10.txt'
    downloader = genetic_testdata(download_path)
    # plink_stem = downloader.download_1kg_chr22()
    plink_stem = downloader.download_ukb_chr10()
    ld_blocks = downloader.download_ldblocks()
    pheno_file = downloader.download_file(sim_path)
    # Models
    models = {'pytorch': pytorch_linear,
              'tensor': tensorflow_models,
              'sklearn': sklearn_models}
    # Phenotype processing
    # pheno_reader = DataProcessing(pheno_file)
    # ph = pheno_reader.get_pheno(1)
    # y = ph['pheno'].values
    y = np.random.random(1092)
    # Reading of genetic data
    genetic_process = Genetic_data_read(plink_stem, ld_blocks)
    out = genetic_process.block_iter(10)
    # Trial run for a single LD block
    for i in out:
        X = i
    # X = scale(X)
    # lamb = 0.01
    # # Setting up the model
    # model_comparision_file = os.path.join(download_path, 'model.comparisions3')
    # for i, m in models.items():
    #     print(i)
    #     pytorchmodel = m(X, y, model_comparision_file,
    #                      False, type='c', mini_batch_size=100)
    #     pytorchmodel.run(penal='l1')
    #     pytorchmodel.run(penal='l2')
