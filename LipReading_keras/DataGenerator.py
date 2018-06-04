# Data Generator

import numpy as np
import keras
from numpy.random import *
import h5py

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, name=None, batch_size=32, file_name=None, data_sampels=None):
        'Initialization'
        self.data_sampels = data_sampels
        self.batch_size = batch_size
        self.name = name
        self.file_name= file_name
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data_sampels / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        X, y = self.__data_generation()

        return X, y

    def __data_generation(self):
        'Generates data containing batch_size samples'
        hdf5_file = h5py.File(self.file_name, mode='r')

        x = np.empty((self.batch_size, 29, 112, 112, 1))
        y = np.empty((self.batch_size,29 , 500))

        for i in range(0, self.batch_size):
            idx = randint(0,self.data_sampels-1)
            if (self.name == 'train'):
                x[i] = hdf5_file['x_' + self.name][idx]
                y[i] = hdf5_file['y_' + self.name][keras.utils.to_categorical(idx, num_classes=500)]

            if (self.name == 'val'):
                x[i] = hdf5_file['x_' + self.name][idx]
                y[i] = hdf5_file['y_' + self.name][keras.utils.to_categorical(idx, num_classes=500)]

        return x, y
