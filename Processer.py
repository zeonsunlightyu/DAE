import numpy as np
import pandas as pd
from scipy.special import erfinv
import re

def drop_columns(df, col_names):
        
    print('Before drop columns {0}'.format(df.shape))
    df = df.drop(col_names, axis = 1)
    print('After drop columns {0}'.format(df.shape))
    return df
   
def dtype_transform(df):
        
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(np.float32)
    return df

def label_encode(df,col_names):
       
    df = preprocessing.LabelEncoder().fit(df[col_names]).transform(df[col_names])
        
    return df

def fill_missing(df,col_names,fill_values):
       
    for i in range(len(col_names)):
        print('filling {0}'.format(col_names[i]))
        df[col_names[i]] = fill_values[i].fit(df[col_names[i]]).transform(df[col_names[i]])
        
    return df
 
def onehot(df,cat_features,drop_origin = True,threshold=0):
        
    for column in cat_features:
        print("the number of unique number(exclude threhold): {0}".format(len(df[column].unique())))
        dummy_column = pd.get_dummies(pd.Series(df[column]), prefix=column)
        abort_columns = []
        for col in dummy_column:
            if dummy_column[col].sum() < threshold:
                print('column {0} unique value {1} less than threshold {2}'.format(col,dummy_column.sum(),threshold))
                abort_columns.append(col)
        print("Abort columns : {0}".format(abort_columns))
        remain_cols = [c for c in dummy_column.columns if c not in abort_columns]
        df = pd.concat([df,dummy_column[remain_cols]],axis=1)
        if drop_origin:
            print("Drop column : {0}".format(column))
            df = df.drop([column], axis=1)
    return df

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
    
    #for v,v_oc in hist.items():
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

def lower_bound(sequence, value, compare=cmp):
    
    elements = len(sequence)
    offset = 0
    middle = 0
    found = len(sequence)
 
    while elements > 0:
        middle = elements / 2
        if compare(value, sequence[offset + middle]) > 0:
            offset = offset + middle + 1
            elements = elements - (middle + 1)
        else:
            found = offset + middle
            elements = middle
    return found

def apply_rankgauss_map(df,trafo_map):
    
    values = df.values
    sorted_key_list = sorted(list(trafo_map.keys()))
    rankgauss_values = []
    
    for index, value in enumerate(values):
        
        rankgauss_value = 0
        if value in trafo_map:
            rankgauss_value = trafo_map[value]       
        else:
            lb = lower_bound(sorted_key_list, value)
            
            #high clip
            if lb > len(sorted_key_list)-1:
                lb_value = sorted_key_list[len(sorted_key_list)-1]
                rankgauss_value = trafo_map[lb_value]
            else:
                #low clip
                lb_value = sorted_key_list[lb]
                rankgauss_value = trafo_map[lb_value]
      
                if lb > 0:
                    x0 = sorted_key_list[lb-1]
                    y0 = trafo_map[x0]
                    x1 = lb_value
                    y1 = trafo_map[x1]
                    rankgauss_value = y0 + (value - x0) * (y1 - y0) / (x1 - x0)
                
        rankgauss_values.append(rankgauss_value)
        
    return np.array(rankgauss_values)

def rank_gauss(df,col_name):
       
    rankgauss_map = build_rankgauss_trafo(df[col_name])
    return apply_rankgauss_map(df[col_name],rankgauss_map)

def perform_rank_gauss(df,col_names):
    df[col_names] = df[col_names].replace(-1,0)
    for c in col_names:
        print(c)
        print(df[c].dtype)
        df[c] = rank_gauss(df,c)      
    return df

class Compose(object):
    def __init__(self, transforms_params):
        self.transforms_params = transforms_params
    def __call__(self, df):
        for transform_param in self.transforms_params:
            transform, param = transform_param[0], transform_param[1]
            df = transform(df, **param)
        return df
