import keras
import os
import pickle
import logging
import numpy as np

lg = logging.getLogger(__name__)




class KerasCombinedModel(object):

    def __init__(self, input_dim: int, ld_block: str):
        self.ld_blocks = pickle.load(open(ld_block, 'rb'))[10]
        self.p = input_dim

    def _make_block_id(self) -> list:
        output = list()
        u = 0
        for i, b in enumerate(self.ld_blocks):
            nn = len(b)
            mask = np.zeros(self.p, dtype=bool)
            mask[u:(u + nn)] = True
            u += nn
            output.append(mask)
            if i % 10 == 0:
                lg.debug('Processing LD block %s', i)
        return output

    def model(self):
        blocks = []
        inputs = list()
        for b in blocks:
            shape = len(b)
            inp = keras.Input(shape=shape)
            inp = keras.layers.Dense(1)(inp)
            inputs.append(inp)

        combined_layer = keras.layers.Concatenate(axis=1)(inputs)
        output = keras.layers.Dense(1)(combined_layer)
