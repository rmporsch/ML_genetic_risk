import logging
import os
import json
import pandas as pd
from spyplink.converting import Converting
from typing import List

lg = logging.getLogger(__name__)


class Processing(Converting):

    def __init__(self, plink_path: str, output_dir: str,
                 pheno_file: str, parameter_path: str, ld_block_path: str = None):
        Converting.__init__(self, plink_path, output_dir, pheno_file,
                            ld_block_path)
        self._ispath(parameter_path)
        self.parameters = json.load(parameter_path)[0]

    def traindev_pipeline(self, rerun: bool = False):
        # TODO: move training and dev temp files to output folder
        train_path = '.train.temp'
        dev_path = '.dev.temp'
        # train_dev_split
        batch_size = self.parameters['batch_size']
        frac = self.parameters['frac']
        if rerun or (os.path.isfile(train_path) and os.path.isfile(dev_path)):
            train = pd.read_table(train_path, header=None, sep='\t')
            dev = pd.read_table(dev_path, header=None, sep='\t')
        else:
            train, dev = self.train_dev_split(batch_size, frac)

        return train, dev

    def gwas_pipeline(self, phenotypes: List):
        pass

    def clumping_pipeline(self):
        pass

    def converting_pipeline(self):
        pass

    def processing_pipeline(self, phenotypes: List, rerun=False):
        train, dev = self.traindev_pipeline(phenotypes, rerun)










