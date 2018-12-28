from keras import optimizers
from keras import losses

noise_level = 0.15
epoch = 1000
batch_size = 128
opt = optimizers.SGD(lr=0.003, momentum=0.0, decay=0.995, nesterov=False)
loss = losses.mean_squared_error
lr = 0.003
wdecay = 0.995
