#Conv-backend
import keras
from keras.layers import *

def Conv_BackEnd(input):
    
    permute = Permute((2, 1))(input)

    conv1 = Conv1D(512,kernel_size= 2 ,strides=2, padding='same', name = 'conv1')(permute)

    BN2 =  BatchNormalization(axis=2, momentum=0.9, epsilon=0.001)(conv1)

    relu2 = Activation("relu")(BN2)

    max_pool1 = MaxPooling1D(pool_size=2, strides=2, padding='same', name = 'max_1')(relu2)

    conv1_sec = Conv1D(1024,kernel_size= 2, strides=2, padding='same')(max_pool1)

    BN3 =  BatchNormalization(axis=2, momentum=0.9, epsilon=0.001)(conv1_sec)

    relu3 = Activation("relu")(BN3)

    avg_pool = GlobalAveragePooling1D()(relu3)

    linear_first = Dense(256, name = 'dense1')(avg_pool)

    BN4 =  BatchNormalization(axis=1, momentum=0.9, epsilon=0.001, name = 'BN')(linear_first)

    relu4 = Activation("relu")(BN4)

    linear_sec = Dense(500, name = 'dense2', activation='linear')(relu4)

    output = Activation("softmax")(linear_sec)
    
    return output