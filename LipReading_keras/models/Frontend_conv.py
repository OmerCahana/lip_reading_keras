# frontEnd model

import keras
from models import resnet
from keras.layers import *
from keras.models import Model

def FrontEnd(inputs):
    
    conv3 = Conv3D(filters = 64, kernel_size=(5, 7, 7), kernel_initializer='glorot_uniform',
                        strides=(1, 2, 2),  padding="same", data_format="channels_last", name = 'conv3d')(inputs)

    BN = BatchNormalization(axis=4, momentum=0.9, epsilon=0.001)(conv3)

    relu = Activation("relu")(BN)

    max_pool = MaxPooling3D(pool_size=(1, 2, 2), strides=None, padding='valid', data_format='channels_last',
                            name = 'max_pooling_first')(relu)

    Resnet  = TimeDistributed(resnet.Build_resnet_34(input_shape =(64,28,28) ,num_outputs = 256))(max_pool)
    
    return Resnet
