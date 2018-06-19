from DataGenerator import DataGenerator
from models import Conv_backend,Frontend_conv,lstm_backend
from keras.models import Model
from keras import optimizers
from keras.layers import *
from NLLoss import *

inputs = Input(shape=(29,112,112,1), dtype='float32', name='main_input')

FrontEnd_output = Frontend_conv.FrontEnd(inputs) 

lstm_output = lstm_backend.backend_lstm(FrontEnd_output)

model = Model(inputs=inputs, outputs =lstm_output)

training_generator = DataGenerator(name = 'train', batch_size = 64,file_name = "train_set.hdf5", data_sampels = 488766)
validation_generator = DataGenerator(name = 'val', batch_size = 64,file_name =  "val_set.hdf5",data_sampels =25000)

sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=False)

model.compile(optimizer=sgd ,
              loss=NLLoss,
              metrics=['accuracy'])

model.fit_generator(generator=training_generator,validation_data=validation_generator,
                    use_multiprocessing=True,workers=18,epochs=5)