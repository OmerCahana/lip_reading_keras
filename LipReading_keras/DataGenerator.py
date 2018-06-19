# Data Generator

import numpy as np
import keras
from numpy.random import *
from Data.Preprocessing import *


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, name=None, batch_size=32, data_sampels=None):
        'Initialization'
        self.data_sampels = data_sampels
        self.batch_size = batch_size
        self.list = build_file_list(directory='/home/omer4436/lipread_mp4', set_name=name)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((self.data_sampels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        X, y = self.__data_generation()

        return X, y

    def __data_generation(self):
        'Generates data containing batch_size samples'

        x = np.empty((self.batch_size, 29, 112, 112, 1))
        y = np.empty((self.batch_size, 29, 500))

        for i in range(0, self.batch_size):
            idx = randint(0, (self.data_sampels) - 1)

            x[i] = video2data(self.list[idx][1])
            y[i] = keras.utils.to_categorical(self.list[idx][0], num_classes=5)

        y = y[:, 0, :]
        return x, y