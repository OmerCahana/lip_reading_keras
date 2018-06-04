# lstm-backend
import keras
from keras.layers import *

def backend_lstm(input):
  
    linear_first = Dense(256)(input)

    bi_dir = Bidirectional(LSTM(256, return_sequences=True, go_backwards = True), merge_mode=None)(linear_first)

    lstm_side1 = LSTM(256, go_backwards = True, return_sequences=True)(bi_dir[0])

    lstm_side2= LSTM(256, return_sequences=True)(bi_dir[1])

    concat_lstm = keras.layers.concatenate([lstm_side1, lstm_side2])

    output = Dense(50, activation = 'softmax')(concat_lstm)
    
    return output