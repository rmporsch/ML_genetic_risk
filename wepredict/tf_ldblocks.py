import numpy as np
import tensorflow as tf
import tensorboard as tb
import logging
from pyplink import PyPlink


def get_genotypes(self, chr, rsid, plink_path, sub_in):
    reader = PyPlink(plink_path)
    n = reader.get_nb_samples()
    genotypematrix = np.zeros((n, len(rsid)), dtype=np.int8)
    pos_index = 0
    for snp, genotype in reader.iter_geno_marker(rsid):
        if snp not in rsid:
            continue
        else:
            genotypematrix[:, pos_index] = genotype[sub_in]
            pos_index += 1
    reader.close()
    return genotypematrix

class NNpredict(object):

    def __init__(self):
        super(NNpredict, self).__init__()






