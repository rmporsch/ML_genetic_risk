import keras
import pickle
from keras import backend as K
import tensorflow as tf
import logging
import numpy as np
from spyplink.plink_reader import Major_reader
from typing import List
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, merge
from keras.layers.merge import concatenate
from keras import regularizers, optimizers

lg = logging.getLogger(__name__)


class DataGenerator(keras.utils.Sequence, Major_reader):

    def __init__(self, plink_file: str, pheno_file, pheno_name: str,
                 batch_size: int, ldblock_file: str = None, shuffle=True):
        """
        Primary data generator for keras

        :param plink_file: path of a plink file in sample major format
        :param pheno_file: path of the pheno file
        :param pheno_name: name of the phenotype
        :param batch_size: size of the mini batches
        :param ldblock_file: path of the ld block file (bed)
        :param shuffle: bool if the data should be shuffled
        """
        Major_reader.__init__(self, plink_file, pheno_file, ldblock_file)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pheno_name = pheno_name
        self.indexes = np.arange(0, self.n, dtype=int)
        self.on_epoch_end()
        self.dims = None
        if ldblock_file is not None:
            self.block_sequence = self._generate_ld_split_sequence()
            # check dims
            tx, ty = self.__getitem__(0)
            self.dims = [k.shape[1] for k in tx]
        else:
            self.block_sequence = None
            self.dims = self.p

    def _generate_ld_split_sequence(self) -> List:
        """
        Generates a sequence of numbers to split the genotype  matrix
        :return: list with split points
        """
        block_sequence = list()
        pos = 0
        for u, k in enumerate(self.chrom):
            lg.debug('Making blocks for chr %s', k)
            for i, b in enumerate(self.ldblocks[k]):
                num = len(b)
                if num == 0:
                    continue
                pos += num
                block_sequence.append(pos)
                if i % 10 == 0:
                    lg.debug('Processed block %s', i)
        del block_sequence[-1]
        lg.debug('Generated block sequence: %s', block_sequence)
        return block_sequence

    def _data_generation(self, list_id_temp, pheno_name):
        """
        Generates a batch size from the data

        :param list_id_temp: list of ids to sample
        :param pheno_name: the name of the phenotype
        :return: (x, y)
        """

        y_iter = self._one_iter_pheno(pheno_name, list_id_temp)
        x_iter = self._one_iter_geno(list_id_temp)

        x = np.empty((self.batch_size, self.p))
        y = np.empty(self.batch_size)

        for i, data in enumerate(zip(x_iter, y_iter)):
            g, p = data
            x[i, ] = g.flatten()
            y[i] = p.flatten()

        if self.block_sequence is not None:
            lg.debug('Dim of x: %s, size of p: %s',
                     x.shape, self.p)
            x = np.split(x, np.array(self.block_sequence), axis=1)
            lg.debug('Size of output: %s', len(x))
        return x, y

    def on_epoch_end(self):
        """
        Shuffle or not on an epoch end
        :return:  None
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(self.n / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        x, y = self._data_generation(indexes, self.pheno_name)
        return x, y


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r


def ld_nn(input_dims: List, drop_r: float = 0.5,
          l_rate: float = 0.001):
    """
    Make model
    :param input_dims: list of input dims
    :param drop_r: rate of dropout
    :param l_rate:  learning rate
    :return: compiled model
    """
    inputs = list()
    mini_model = list()
    for b in input_dims:
        if b == 0:
            continue
        imp = Input(shape=(b,))
        temp = Dense(1,
                     kernel_regularizer=regularizers.l1(0.01))(imp)
        inputs.append(imp)
        mini_model.append(temp)

    combined_layer = concatenate(mini_model)
    model = Dense(20,
                  activation='relu',
                  kernel_regularizer=regularizers.l1(0.01))(combined_layer)
    output = Dense(1, activation='linear')(model)

    final_model = Model(inputs, output=output)

    opti = optimizers.Adagrad(lr=l_rate)
    final_model.compile(loss='mse', optimizer=opti,
                  metrics=[correlation_coefficient_loss],)
    return final_model

def nnmodel(input_n: int, drop_r: float = 0.5,
            l_rate: float = 0.001):
    model = Sequential()
    model.add(Dense(units=85, activation='elu',
                    input_dim=input_n,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(drop_r))
    model.add(Dense(units=60, activation='elu',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(drop_r))
    model.add(Dense(units=60, activation='elu',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(drop_r))
    model.add(Dense(units=60, activation='elu',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dense(units=1, activation='linear',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l1(0.01)))
    opti = optimizers.Adagrad(lr=l_rate)
    model.compile(loss='mse', optimizer=opti,
                  metrics=[correlation_coefficient_loss], )
    return model

def linear_model(input_n: int, l_rate=0.01):
    model = Sequential()
    model.add(Dense(units=1, activation='linear',
                    input_dim=input_n,
                    kernel_regularizer=regularizers.l1(0.01)))
    opti = optimizers.Adagrad(lr=l_rate)
    model.compile(loss='mse', optimizer=opti,
                  metrics=[correlation_coefficient_loss], )
    return model

def nnmodel_small(input_n: int, l_rate: float = 0.001):
    model = Sequential()
    model.add(Dense(units=15, activation='elu',
                    input_dim=input_n,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dense(units=1, activation='linear',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l1(0.01)))
    opti = optimizers.Adagrad(lr=l_rate)
    model.compile(loss='mse', optimizer=opti,
                  metrics=[correlation_coefficient_loss], )
    return model
