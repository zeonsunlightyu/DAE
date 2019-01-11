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
kernel_initializer_0=initializers.RandomNormal(mean=-4.74564e-05, stddev=0.0388202, seed=None)   
kernel_initializer_1=initializers.RandomNormal(mean=8.51905e-06, stddev=0.0148989, seed=None)   
kernel_initializer_2=initializers.RandomNormal(mean=8.51905e-06, stddev=0.0148989, seed=None)    
kernel_initializer_3=initializers.RandomNormal(mean=-1.80977e-05, stddev=0.0149005, seed=None)   
metric = 'mse'
