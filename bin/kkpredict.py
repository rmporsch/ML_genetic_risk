import logging
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers, optimizers
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import argparse
from keras.callbacks import TensorBoard
from nnpredict.kk_model import DataGenerator
from nnpredict.kk_model import ld_nn, correlation_coefficient_loss
from nnpredict.kk_model import nnmodel, nnmodel_small, linear_model


par = argparse.ArgumentParser(description='Convert plink files.')

par.add_argument('model', type=str,
                 help='model type')
par.add_argument('-p', type=int, default=10, help='number of workers')

args = par.parse_args()

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    print(os.getcwd())
    ldblocks = 'data/Berisa.EUR.hg19.bed'
    train_path = 'data/sample_major/1kg/clumped/sim_1000G_chr10_SampleMajor_train'
    dev_path = 'data/sample_major/1kg/clumped/sim_1000G_chr10_SampleMajor_dev'
    # train_path = 'data/sample_major/ukb/clumped/nonlinear/maf_0.01_10_SampleMajortrain'
    # dev_path = 'data/sample_major/ukb/clumped/nonlinear/maf_0.01_10_SampleMajordev'
    pheno_path = 'data/sim_1000G_chr10.txt'
    # pheno_path = 'data/simulated_chr10.txt'
    # pheno_path = 'data/pseudophenos_mini.txt'
    var = 'V1'
    m = args.model
    print(m)

    batch_size = 1000
    if '1000G' in train_path:
        batch_size = 100
    print(batch_size)

    adpar = list()
    if m == 'ld_nn':
        adpar = [ldblocks]

    train_generator = DataGenerator(train_path, pheno_path, var,
                                    batch_size, *adpar)
    dev_generator = DataGenerator(dev_path, pheno_path, var,
                                  batch_size, *adpar)
    p = train_generator.dims

    models = {'linear': linear_model, 'nn_small': nnmodel_small,
              'nn': nnmodel, 'ld_nn': ld_nn}
    model = models[m](p)
    # print(model.summary())
    opti = optimizers.Adagrad(lr=0.001)
    model.compile(loss='mse', optimizer=opti,
                  metrics=[correlation_coefficient_loss],)
    tensorboard = TensorBoard(log_dir='./.tb_'+m)

    # Train model on dataset
    history = model.fit_generator(generator=train_generator,
                                  validation_data=dev_generator,
                                  use_multiprocessing=True,
                                  workers=args.p, epochs=100,
                                  callbacks=[tensorboard])

    # summarize history for accuracy
    dump_name = 'non_linear_keras_model_uk_'+m+'.pickle'
    pickle.dump(history, open(dump_name, 'wb'))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['correlation_coefficient_loss'], label='train')
    ax.plot(history.history['val_correlation_coefficient_loss'], label='test')
    ax.set_title('Model Accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(loc='upper right')
    fig_name = 'non_linear_keras_model_uk_'+m+'.png'
    fig.savefig(fig_name)
