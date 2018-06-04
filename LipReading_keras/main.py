from DataGenerator import DataGenerator
import keras
from models import Conv_backend,Frontend_conv,lstm_backend
from keras.utils import multi_gpu_model
from keras.models import Model
from keras import optimizers
from keras.layers import *
from NLLoss import *

inputs = Input(shape=(29,112,112,1), dtype='float32', name='main_input')

FrontEnd_output = Frontend_conv.FrontEnd(inputs) 

lstm_output = lstm_backend.backend_lstm(FrontEnd_output)

model = Model(inputs=inputs, outputs =lstm_output)

training_generator = DataGenerator(name = 'train', batch_size = 64,file_name = "train_set.hdf5", data_sampels = 400000)
validation_generator = DataGenerator(name = 'val', batch_size = 64,file_name =  "val_set.hdf5",data_sampels =25000)

adam = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)

model.compile(optimizer=adam,
              loss=NLLoss,
              metrics=['accuracy'])

model.fit_generator(generator=training_generator,validation_data=validation_generator,
                    use_multiprocessing=True,workers=18,epochs=5)