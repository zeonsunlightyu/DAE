import numpy as np

def rational_approximation(t):
    
    C = [2.515517, 0.802853, 0.010328]
    D = [1.432788, 0.189269, 0.001308]
    result = t - ((C[2]*t + C[1])*t + C[0]) / (((D[2]*t + D[1])*t + D[0])*t + 1.0)
    
    return result

def normal_CDF_inverse(p):
    
    if (p <= 0.0 or p >= 1.0):
        return False
    if (p < 0.5):
        return -rational_approximation(np.sqrt(-2.0*np.log(p)))
    return rational_approximation(np.sqrt(-2.0*np.log(1-p)))

def erfinv(x):
    
    if (x == 0.0):
        return 0.0
    else:
        if (x < 0.0):
            return -normal_CDF_inverse(-x)*0.7
        else:
            return normal_CDF_inverse(x)*0.7

def build_rankgauss_trafo(df):
    
    trafo_map = {}
    values = df.values
    mean = 0.0
    count = 0
    N = 0

    #build historgram feature-wise
    hist = dict(df.value_counts())
    hist_sorted_list = sorted(hist.items(), key=lambda x: x[0])
    
    for v,v_oc in hist.iteritems():
        N = N + v_oc
        
    for v,v_oc in hist_sorted_list:

        rank_v = float(count)/float(N)
   
        rank_v = rank_v*0.998 + 1e-3
        rank_v = erfinv(rank_v)
        
        mean = mean + v_oc*rank_v
        
        trafo_map[v] = rank_v

        count = count + v_oc
        
    mean = mean/float(N)
    
    for v,rank_v in trafo_map.iteritems():
        trafo_map[v] = rank_v - mean
    
    return trafo_map

def apply_rank_trafo(df,trafo_map):
    
    
