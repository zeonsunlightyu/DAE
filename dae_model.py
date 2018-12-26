from utils import *
from keras.models import Model
from keras.layers import Dense,Input,Activation
from keras.layers.advanced_activations import PReLU
from keras import optimizers
from keras import losses

def nn_model():
    
    input = Input(shape = (221,), dtype='float32')
    
    fc1_encode = Dense(1500,init = 'he_normal',activation='relu')(input)
    fc_bottle_neck = Dense(1500,init = 'he_normal',activation='relu')(fc1_encode)
    fc1_decode = Dense(1500,init = 'he_normal',activation='relu')(fc_bottle_neck)
    
    output = Dense(221,activation='linear')(fc1_decode)
    
    sgd = optimizers.SGD(lr=0.003, decay=0.995)
    model = Model(input = input, output = output)
    model.compile(loss = losses.mean_squared_error,
                  optimizer = sgd)
    
    return model
  

    
    
  

