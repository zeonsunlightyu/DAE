import numpy as np

def add_swap_noise(cur_data, all_data, noise_level=0.15):
    """
    copy from randxie's blog
    Add swap noise to current data
    :param cur_data: Current batch of data
    :param all_data: The whole data set
    :param noise_level: percentage of columns being swapped
    :return: data with swap noise added
    """
    batch_size = cur_data.shape[0]
    num_samples = all_data.shape[0]
    num_features = cur_data.shape[1]
    #可能会导致同一个batch内部的互换,虽然概率应该很低
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
            #reset random seed
            #np.random.seed(np.random.randint(1000000000))
            
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
