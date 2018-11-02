import logging
import os
from pyplink_major.plink_reader import Major_reader
import numpy as np
import keras
from keras import backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
from keras import regularizers
from keras import optimizers
from matplotlib import pyplot as plt


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
    return K.square(r)


if __name__ == '__main__':
    os.chdir('/home/robert/Documents/projects/ML_genetic_risk')
    print(os.getcwd())
    train_path = 'data/sample_major/1kg/clumped/sim_1000G_chr10_SampleMajor_train'
    dev_path = 'data/sample_major/1kg/clumped/sim_1000G_chr10_SampleMajor_dev'
    pheno_path = 'data/sim_1000G_chr10.txt'
    plink_path = 'data/sim_1000G_chr10'
    var = 'V1'

    train_generator = DataGenerator(train_path, pheno_path, var, 100)
    dev_generator = DataGenerator(dev_path, pheno_path, var, 50)

    model = Sequential()
    model.add(Dense(units=60, activation='relu',
                    input_dim=116,
                    kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(units=60, activation='relu',
                    kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='linear',
                    kernel_regularizer=regularizers.l1(0.01)))
    opti = optimizers.Adagrad(lr=0.01)
    model.compile(loss='mse', optimizer=opti,
                  metrics=[correlation_coefficient_loss], )

    # Train model on dataset
    history =  model.fit_generator(generator=train_generator,
                        validation_data=dev_generator,
                        use_multiprocessing=True,
                        workers=2, epochs=100)

    # summarize history for accuracy
    plt.plot(history.history['correlation_coefficient_loss'], label='train')
    plt.plot(history.history['val_correlation_coefficient_loss'], label='test')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.savefig('keras_model.png')
    plt.show()






