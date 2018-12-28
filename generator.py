import numpy as np

def add_swap_noise(cur_data, all_data, noise_level=0.15):
  
    batch_size = cur_data.shape[0]
    num_samples = all_data.shape[0]
    num_features = cur_data.shape[1]

    random_row = np.random.randint(0, num_samples, size=batch_size)
    for i in range(batch_size):
        random_swap = np.random.rand(num_features) < noise_level
        cur_data[i, random_swap] = all_data[random_row[i], random_swap]
    return cur_data

def batch_generator(data, batch_size, shuffle=True):
    
    batch_index = 0
    n = data.shape[0]
    
    while True:
        
        if batch_index == 0:
            index_array = np.arange(n)
            if shuffle:
                index_array = np.random.permutation(n)
            
        current_index = (batch_index * batch_size) % n
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = n - current_index
            batch_index = 0

        batch = data[index_array[current_index: current_index + current_batch_size]]
        
        yield batch

def swap_noise_generator(data,batch_size,shuffle=True,noise_level=0.15):
    
    gen = batch_generator(data, batch_size, shuffle)
    while True:
        batch = next(gen)
        y = batch.copy()
        x = add_swap_noise(batch, data, noise_level=noise_level)
        
        yield (x,y)
