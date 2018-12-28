from config import *
from keras.models import Model
from keras.layers import Dense,Input,Activation
from keras.layers.advanced_activations import PReLU
from keras import optimizers
from keras import losses
from generator import *

class NN_model(Object):
    def __init__(self,X):
        self.noise_level = config.noise_level
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.opt = config.opt
        self.loss = config.loss
        self.lr = config.lr
        self.wdecay = config.wdecay
        self.X = X
        self.model = self.bulid_model()
        self.data_generator = swap_noise_generator(data = self.X,batch_szie = )
        
    def build_model(self):
        
        inputs = Input((self.X.shape[1],))
        x = Dense(1500, activation='relu')(inputs)
        x = Dense(1500, activation='relu')(x)
        x = Dense(1500, activation='relu')(x)
        outputs = Dense(self.X.shape[1], activation='relu')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.opt, loss=self.loss)
        
        return model
    
    def fit(self):
        
        self.model.fit_generator(generator=self.data_generator,
                  steps_per_epoch=np.ceil(self.X.shape[0] / self.batch_size),
                  epochs=self.epoch,
                  verbose=1)
