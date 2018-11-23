#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from keras.callbacks import TensorBoard, ModelCheckpoint
from nnpredict.kk_model import DataGenerator
from nnpredict.kk_model import ld_nn, correlation_coefficient_loss
from nnpredict.kk_model import nnmodel, nnmodel_small, linear_model
from glob2 import glob


par = argparse.ArgumentParser(description='Convert plink files.')

par.add_argument('train', type=str, help='path to train file')
par.add_argument('dev', type=str, help='path to dev file')
par.add_argument('pheno', type=str, help="path to pheno")
par.add_argument('pname', type=str, help='pheno name')
par.add_argument('output', type=str, help='outputname')

par.add_argument('-p', type=int, default=10, help='number of workers')

par.add_argument("-v", "--verbose", action="store_const",
                 dest="log_level", const=logging.INFO,
                 default=logging.WARNING)

par.add_argument("-d", "--debug",
                 action="store_const", dest="log_level",
                 const=logging.DEBUG)

args = par.parse_args()
logging.basicConfig(level=args.log_level)
lg = logging.getLogger(__name__)


if __name__ == '__main__':
    print(os.getcwd())
    ldblocks = 'data/Berisa.EUR.hg19.bed'
    train = args.train
    dev = args.dev
    pheno = args.pheno
    batch_size = 1000
    adpar = list()
    var = args.pname

    train_generator = DataGenerator(args.train, args.pheno, var,
                                    batch_size, *adpar)
    dev_generator = DataGenerator(args.dev, args.pheno, var,
                                  batch_size, *adpar)
    p = train_generator.dims

    models = {'linear': linear_model, 'nn_small': nnmodel_small,
              'nn': nnmodel, 'ld_nn': ld_nn}
    model = models['nn_small'](p)
    tensorboard = TensorBoard(log_dir='./.tb/height_'+args.output)
    checkpoint = ModelCheckpoint(args.output+'.model.{epoch:02d}_.checkpoint',
            monitor='val_correlation_coefficient_loss',
            mode='max', save_best_only=True, verbose=1)

    # Train model on dataset
    history = model.fit_generator(generator=train_generator,
                                  validation_data=dev_generator,
                                  use_multiprocessing=True,
                                  workers=args.p, epochs=50,
                                  callbacks=[tensorboard, checkpoint])

    dev_generator = DataGenerator(args.dev, args.pheno, var,
                                  1, shuffle=False)
    prediction = model.predict_generator(dev_generator, workers=args.p, use_multiprocessing=True)
    pred_dump = args.output+'_pred.pickle'
    pickle.dump(prediction, open(pred_dump, 'wb'))
    model.save(args.output+'_model.keras.h5')

    # summarize history for accuracy
    dump_name = args.output+'_model.pickle'
    pickle.dump(history, open(dump_name, 'wb'))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['correlation_coefficient_loss'], label='train')
    ax.plot(history.history['val_correlation_coefficient_loss'], label='test')
    ax.set_title('Model Accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend()
    fig_name = args.output+'.png'
    fig.savefig(fig_name)
