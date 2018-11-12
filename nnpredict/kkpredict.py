import logging
import os
from spyplink.plink_reader import Major_reader
import numpy as np
import keras
from keras import backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
from keras import regularizers
from keras import optimizers
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle
import pandas as pd


class DataGenerator(keras.utils.Sequence, Major_reader):

    def __init__(self, plink_file: str, pheno_file, pheno_name,
                 batch_size, shuffle=True):
        Major_reader.__init__(self, plink_file, pheno_file)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pheno_name = pheno_name
        self.indexes = np.arange(0, self.n, dtype=int)
        self.on_epoch_end()

    def _data_generation(self, list_id_temp, pheno_name):

        y_iter = self._one_iter_pheno(pheno_name, list_id_temp)
        x_iter = self._one_iter_geno(list_id_temp)

        x = np.empty((self.batch_size, self.p))
        y = np.empty(self.batch_size)

        for i, data in enumerate(zip(x_iter, y_iter)):
            g, p = data
            x[i, ] = g.flatten()
            y[i] = p.flatten()

        return x, y

    def on_epoch_end(self):
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


if __name__ == '__main__':
    os.chdir('/home/rmporsch/projects/ML_genetic_risk')
    print(os.getcwd())
    # train_path = 'data/sample_major/1kg/clumped/sim_1000G_chr10_SampleMajor_train'
    # dev_path = 'data/sample_major/1kg/clumped/sim_1000G_chr10_SampleMajor_dev'
    train_path = 'data/sample_major/ukb/clumped/maf_0.01_10_SampleMajortrain'
    dev_path = 'data/sample_major/ukb/clumped/maf_0.01_10_SampleMajordev'
    # pheno_path = 'data/sim_1000G_chr10.txt'
    pheno_path = 'data/simulated_chr10.txt'
    # pheno_path = 'data/pseudophenos_mini.txt'
    var = 'V1'

    bim = pd.read_table(dev_path+'.bim', header=None)
    p = bim.shape[0]
    batch_size = 1000
    if '1000G' in train_path:
        batch_size = 100
    print(batch_size)
    print(p)

    train_generator = DataGenerator(train_path, pheno_path, var, batch_size)
    dev_generator = DataGenerator(dev_path, pheno_path, var, batch_size)


    def nnmodel(input_n: int = 274):
        model = Sequential()
        model.add(Dense(units=85, activation='elu',
                        input_dim=input_n,
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l1(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(units=60, activation='elu',
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l1(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(units=60, activation='elu',
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l1(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(units=60, activation='elu',
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l1(0.01)))
        model.add(Dense(units=1, activation='linear',
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l1(0.01)))
        opti = optimizers.Adagrad(lr=0.001)
        model.compile(loss='mse', optimizer=opti,
                      metrics=[correlation_coefficient_loss], )
        return model
    
    def linear_model(input_n: int = 274):
        model = Sequential()
        model.add(Dense(units=1, activation='linear',
                        input_dim=input_n,
                        kernel_regularizer=regularizers.l1(0.01)))
        opti = optimizers.Adagrad(lr=0.01)
        model.compile(loss='mse', optimizer=opti,
                      metrics=[correlation_coefficient_loss], )
        return model

    def nnmodel_small(input_n: int = 274):
        model = Sequential()
        model.add(Dense(units=15, activation='elu',
                        input_dim=input_n,
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l1(0.01)))
        model.add(Dense(units=1, activation='linear',
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l1(0.01)))
        opti = optimizers.Adagrad(lr=0.001)
        model.compile(loss='mse', optimizer=opti,
                      metrics=[correlation_coefficient_loss], )
        return model
    

    # model = linear_model(p)
    model = nnmodel_small(p)

    # Train model on dataset
    history =  model.fit_generator(generator=train_generator,
                        validation_data=dev_generator,
                        use_multiprocessing=True,
                        workers=9, epochs=100)

    # summarize history for accuracy
    pickle.dump(history, open('keras_model_ukb_nn_small_15.pickle', 'wb'))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['correlation_coefficient_loss'], label='train')
    ax.plot(history.history['val_correlation_coefficient_loss'], label='test')
    ax.set_title('Model Accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(loc='upper right')
    fig.savefig('keras_model_ukb_nn_small_15.png')

