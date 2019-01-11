from keras import optimizers
from keras import losses
from keras import initializers
from math import *
noise_level = 0.07
epoch = 1000
batch_size = 128
lr = 0.003
wdecay = 0.995
opt = optimizers.SGD(lr=lr,decay=wdecay)#,momentum=0.9, nesterov=True)
loss = 'mean_squared_error'
init = initializers.RandomUniform(minval=-sqrt(1.0/221), maxval=sqrt(1.0/221), seed=12345)
metric = 'mse'
