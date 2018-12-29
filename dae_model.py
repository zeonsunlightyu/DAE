import config
from keras.models import Model
from keras.layers import Dense,Input,Activation
from keras.layers.advanced_activations import PReLU
from keras import optimizers
from keras import losses
from generator import *

class NN_model:
    def __init__(self,X):
        self.metric = config.metric
        self.noise_level = config.noise_level
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.opt = config.opt
        self.loss = config.loss
        self.lr = config.lr
        self.wdecay = config.wdecay
        self.X = X
        self.init = config.init
        self.data_generator = swap_noise_generator(data = self.X,
                                                   batch_size = self.batch_size,
                                                   noise_level = self.noise_level)
        
    def build_model(self):
        
        inputs = Input((self.X.shape[1],))
        x = Dense(1500, activation='relu',kernel_initializer=self.init)(inputs)
        x = Dense(1500, activation='relu',kernel_initializer=self.init)(x)
        x = Dense(1500, activation='relu',kernel_initializer=self.init)(x)
        outputs = Dense(self.X.shape[1],kernel_initializer=self.init)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.opt, loss=self.loss, metrics=[self.metric])
        print(model.get_weights())
        print(model.summary())
        
        return model
    
    def fit_generator(self):
        self.model = self.build_model()
        self.model.fit_generator(generator=self.data_generator,
                  steps_per_epoch=np.ceil(self.X.shape[0] / self.batch_size),
                  epochs=self.epoch,
                  verbose=1)
    
    def fit(self):
        self.model = self.build_model()
        self.model.fit(x = self.X,
                  y = self.X,
                  batch_size = self.batch_size,
                  epochs=self.epoch,
                  verbose=1)
